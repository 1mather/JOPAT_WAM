"""JOPAT joint diffusion model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import rearrange

from models.common.adaln_attention import AdaLNAttentionBlock, AdaLNFinalLayer
from models.common.utils import SinusoidalPosEmb, init_weights
from models.jopat.obs_encoder import JOPATObservationEncoder


class MultiViewVideoPatchifier(nn.Module):
    """Patchify multi-view videos into tokens."""

    def __init__(
        self,
        num_views: int,
        input_shape: tuple[int, ...] = (8, 224, 224),
        patch_shape: tuple[int, ...] = (2, 8, 8),
        num_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_views = num_views
        iT, iH, iW = input_shape
        pT, pH, pW = patch_shape
        self.T, self.H, self.W = iT // pT, iH // pH, iW // pW
        self.pT, self.pH, self.pW = pT, pH, pW

        self.patch_encoder = nn.Conv3d(
            in_channels=num_chans,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        self.patch_decoder = nn.Linear(embed_dim, num_chans * pT * pH * pW)

    def forward(self, imgs):
        return self.patchify(imgs)

    def patchify(self, imgs):
        imgs = rearrange(imgs, "b v c t h w -> (b v) c t h w")
        feats = self.patch_encoder(imgs)
        feats = rearrange(feats, "(b v) c t h w -> b (v t h w) c", v=self.num_views)
        return feats

    def unpatchify(self, feats):
        imgs = self.patch_decoder(feats)
        imgs = rearrange(
            imgs,
            "b (v t h w) (c pt ph pw) -> b v c (t pt) (h ph) (w pw)",
            v=self.num_views,
            t=self.T,
            h=self.H,
            w=self.W,
            pt=self.pT,
            ph=self.pH,
            pw=self.pW,
        )
        return imgs

    @property
    def num_patches(self):
        return self.num_views * self.T * self.H * self.W


class QueryBasedTrackPatchifier(nn.Module):
    """Patchify tracks together with their fixed query coordinates."""

    def __init__(
        self,
        track_len: int,
        num_points: int,
        track_shape: tuple[int, ...] = (19, 25, 25),
        patch_shape: tuple[int, ...] = (2, 5, 5),
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_points = num_points
        self.track_len = track_len

        iT, iH, iW = track_shape
        pT, pH, pW = patch_shape

        assert iH * iW == num_points, f"Grid size {iH}x{iW} = {iH*iW} != num_points {num_points}"

        self.pad_t = (pT - (iT % pT)) % pT if iT % pT != 0 else 0
        self.pad_h = (pH - (iH % pH)) % pH if iH % pH != 0 else 0
        self.pad_w = (pW - (iW % pW)) % pW if iW % pW != 0 else 0

        padded_T = iT + self.pad_t
        padded_H = iH + self.pad_h
        padded_W = iW + self.pad_w

        self.T, self.H, self.W = padded_T // pT, padded_H // pH, padded_W // pW
        self.iT, self.iH, self.iW = iT, iH, iW
        self.pT, self.pH, self.pW = pT, pH, pW

        self.patch_encoder = nn.Conv3d(
            in_channels=4,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )

        self.patch_decoder = nn.Linear(embed_dim, 2 * pT * pH * pW)

        self.visibility_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, pT * pH * pW),
        )

    def forward(self, tracks, query_positions):
        """Encode tracks of shape ``(B, T, N, 2)`` as patch tokens."""
        return self.patchify(tracks, query_positions)

    def patchify(self, tracks, query_positions):
        """Convert query-conditioned tracks to a patch-token sequence."""
        B, T, N, C = tracks.shape
        assert N == self.num_points, f"Expected {self.num_points} points, got {N}"
        assert C == 2, f"Expected 2D coordinates, got {C}D"
        assert query_positions.shape == (B, N, 2), (
            f"Query positions shape mismatch: {query_positions.shape}"
        )

        queries = query_positions[:, None, :, :].expand(B, T, N, 2)

        tracks_with_query = torch.cat([queries, tracks], dim=-1)

        tracks_grid = rearrange(
            tracks_with_query,
            "b t (h w) c -> b c t h w",
            h=self.iH,
            w=self.iW
        )

        if self.pad_t > 0 or self.pad_h > 0 or self.pad_w > 0:
            tracks_grid = F.pad(
                tracks_grid,
                (0, self.pad_w, 0, self.pad_h, 0, self.pad_t),
                mode='replicate'
            )

        feats = self.patch_encoder(tracks_grid)
        feats = rearrange(feats, "b c t h w -> b (t h w) c")

        return feats

    def unpatchify(self, feats):
        """Decode patch features into tracks of shape ``(B, T, N, 2)``."""
        patches = self.patch_decoder(feats)

        tracks = rearrange(
            patches,
            "b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)",
            t=self.T,
            h=self.H,
            w=self.W,
            c=2,
            pt=self.pT,
            ph=self.pH,
            pw=self.pW,
        )
        if self.pad_t > 0 or self.pad_h > 0 or self.pad_w > 0:
            tracks = tracks[
                :, :,
                :self.iT,
                :self.iH,
                :self.iW,
            ]

        tracks = rearrange(tracks, "b c t h w -> b t (h w) c")

        return tracks

    def predict_visibility(self, feats):
        """Predict per-point visibility from patch features."""
        vis_logits = self.visibility_head(feats)

        vis_logits = rearrange(
            vis_logits,
            "b (t h w) (pt ph pw) -> b (t pt) (h ph w pw)",
            t=self.T,
            h=self.H,
            w=self.W,
            pt=self.pT,
            ph=self.pH,
            pw=self.pW,
        )

        if self.pad_t > 0 or self.pad_h > 0 or self.pad_w > 0:
            vis_logits = vis_logits[:, :self.iT, :self.num_points]

        return vis_logits

    @property
    def num_patches(self):
        """Return the number of track patches."""
        return self.T * self.H * self.W


class TripleTimestepEncoder(nn.Module):
    """Encode independent action, observation, and track timesteps."""

    def __init__(self, embed_dim: int = 512, mlp_ratio: float = 4.0):
        super().__init__()
        self.sinusoidal_pos_emb = SinusoidalPosEmb(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t1, t2, t3):
        temb1 = self.sinusoidal_pos_emb(t1)
        temb2 = self.sinusoidal_pos_emb(t2)
        temb3 = self.sinusoidal_pos_emb(t3)
        temb = torch.cat([temb1, temb2, temb3], dim=-1)
        return self.proj(temb)


class TripleNoisePredictionNet(nn.Module):
    """Predict noise jointly for actions, observations, and tracks."""

    def __init__(
        self,
        global_cond_dim: int,
        image_shape: tuple[int, ...],
        patch_shape: tuple[int, ...],
        num_chans: int,
        num_views: int,
        action_len: int,
        action_dim: int,
        track_len: int,
        num_points: int,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        num_registers: int = 8,
    ):
        super().__init__()
        self.num_points = num_points
        self.track_len = track_len

        self.obs_patchifier = MultiViewVideoPatchifier(
            num_views=num_views,
            input_shape=image_shape,
            patch_shape=patch_shape,
            num_chans=num_chans,
            embed_dim=embed_dim,
        )
        obs_len = self.obs_patchifier.num_patches

        hidden_dim = int(max(action_dim, embed_dim) * mlp_ratio)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        grid_size = int(num_points ** 0.5)
        assert grid_size ** 2 == num_points, f"num_points {num_points} must be perfect square"

        self.track_patchifier = QueryBasedTrackPatchifier(
            track_len=track_len,
            num_points=num_points,
            track_shape=(track_len, grid_size, grid_size),
            patch_shape=(2, 5, 5),
            embed_dim=embed_dim,
        )
        track_token_len = self.track_patchifier.num_patches

        self.timestep_embedding = TripleTimestepEncoder(timestep_embed_dim)

        self.registers = nn.Parameter(
            torch.empty(1, num_registers, embed_dim).normal_(std=0.02)
        )

        total_len = action_len + obs_len + track_token_len + num_registers
        self.pos_embed = nn.Parameter(
            torch.empty(1, total_len, embed_dim).normal_(std=0.02)
        )

        cond_dim = global_cond_dim + timestep_embed_dim
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)

        self.action_inds = (0, action_len)
        self.next_obs_inds = (action_len, action_len + obs_len)
        self.track_inds = (
            action_len + obs_len,
            action_len + obs_len + track_token_len
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(init_weights)

        w = self.obs_patchifier.patch_encoder.weight.data
        nn.init.normal_(w.view([w.shape[0], -1]), mean=0.0, std=0.02)
        nn.init.constant_(self.obs_patchifier.patch_encoder.bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(
        self,
        global_cond,
        action,
        action_t,
        next_obs,
        next_obs_t,
        tracks,
        track_t,
        query_positions,
    ):
        """Predict noise and visibility for the three input modalities."""
        action_embed = self.action_encoder(action)
        next_obs_embed = self.obs_patchifier(next_obs)
        track_embed = self.track_patchifier(tracks, query_positions)

        if len(action_t.shape) == 0:
            action_t = action_t.expand(action.shape[0]).to(
                dtype=torch.long, device=action.device
            )
        if len(next_obs_t.shape) == 0:
            next_obs_t = next_obs_t.expand(next_obs.shape[0]).to(
                dtype=torch.long, device=next_obs.device
            )
        if len(track_t.shape) == 0:
            track_t = track_t.expand(tracks.shape[0]).to(
                dtype=torch.long, device=tracks.device
            )
        temb = self.timestep_embedding(action_t, next_obs_t, track_t)

        registers = self.registers.expand(next_obs.shape[0], -1, -1)
        x = torch.cat((action_embed, next_obs_embed, track_embed, registers), dim=1)
        x = x + self.pos_embed
        cond = torch.cat((global_cond, temb), dim=-1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.head(x, cond)

        action_noise_pred = x[:, self.action_inds[0] : self.action_inds[1]]
        next_obs_noise_pred = x[:, self.next_obs_inds[0] : self.next_obs_inds[1]]
        track_noise_pred = x[:, self.track_inds[0] : self.track_inds[1]]

        action_noise_pred = self.action_decoder(action_noise_pred)
        next_obs_noise_pred = self.obs_patchifier.unpatchify(next_obs_noise_pred)

        track_noise_pred_coords = self.track_patchifier.unpatchify(track_noise_pred)
        track_regression_pred_visibility = (
            self.track_patchifier.predict_visibility(track_noise_pred)
        )

        return (
            action_noise_pred,
            next_obs_noise_pred,
            track_noise_pred_coords,
            track_regression_pred_visibility,
        )


def generate_query_positions(num_points: int, device='cpu', method='grid'):
    """Generate normalized grid or random query positions."""
    if method == 'grid':
        grid_size = int(num_points ** 0.5)
        assert grid_size ** 2 == num_points, "num_points must be perfect square for grid"

        x = torch.linspace(0, 1, grid_size, device=device)
        y = torch.linspace(0, 1, grid_size, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        query_positions = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    elif method == 'random':
        query_positions = torch.rand(num_points, 2, device=device)

    else:
        raise ValueError(f"Unknown method: {method}")

    return query_positions


class JOPAT(nn.Module):
    """Jointly diffuse actions, future RGB observations, and point tracks."""

    def __init__(
        self,
        action_len: int,
        action_dim: int,
        track_len: int,
        num_points: int,
        obs_encoder: JOPATObservationEncoder,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        latent_patch_shape: tuple[int, ...] = (2, 4, 4),
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        num_registers: int = 8,
        num_train_steps: int = 100,
        num_inference_steps: int = 10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        use_track_visibility: bool = False,
        coord_loss_weight: float = 1.0,
        vis_loss_weight: float = 0.5,
    ):
        """Initialize JOPAT."""
        super().__init__()

        self.coord_loss_weight = coord_loss_weight
        self.vis_loss_weight = vis_loss_weight
        self.action_len = action_len
        self.action_dim = action_dim
        self.action_shape = (action_len, action_dim)
        self.track_len = track_len
        self.num_points = num_points
        self.track_shape = (track_len, num_points, 2)
        self.use_track_visibility = use_track_visibility

        self.obs_encoder = obs_encoder
        self.latent_img_shape = self.obs_encoder.latent_img_shape()

        global_cond_dim = self.obs_encoder.feat_dim()
        image_shape = self.latent_img_shape[2:]
        num_views, num_chans = self.latent_img_shape[:2]
        self.noise_pred_net = TripleNoisePredictionNet(
            global_cond_dim=global_cond_dim,
            image_shape=image_shape,
            patch_shape=latent_patch_shape,
            num_chans=num_chans,
            num_views=num_views,
            action_len=action_len,
            action_dim=action_dim,
            track_len=track_len,
            num_points=num_points,
            embed_dim=embed_dim,
            timestep_embed_dim=timestep_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            num_registers=num_registers,

        )

        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )


    def forward(
        self,
        obs_dict,
        next_obs_dict,
        action,
        tracks,
        action_mask=None
    ):
        """Compute the joint diffusion training loss."""

        batch_size = action.shape[0]
        device = action.device

        obs, next_obs = self.obs_encoder.encode_curr_and_next_obs(
            obs_dict, next_obs_dict
        )

        if tracks is not None and tracks.shape[-1] == 3:
            track_coords = tracks[..., :2]
            track_vis_gt = tracks[..., 2]
        elif tracks is not None and tracks.shape[-1] == 2:
            track_coords = tracks
            track_vis_gt = None
        else:
            shape = None if tracks is None else tuple(tracks.shape)
            raise ValueError(f"Expected tracks ending in 2 or 3 values, got {shape}.")

        coords_noise = torch.randn_like(track_coords)
        track_t = torch.randint(
            low=0,
            high=self.num_train_steps,
            size=(batch_size,),
            device=device
        ).long()

        noisy_coords = self.noise_scheduler.add_noise(
            track_coords,
            coords_noise,
            track_t
        )
        noisy_tracks_input = noisy_coords

        action_noise = torch.randn_like(action)
        action_t = torch.randint(0, self.num_train_steps, (batch_size,), device=device).long()
        if action_mask is not None:
            action_t[~action_mask] = self.num_train_steps - 1
        noisy_action = self.noise_scheduler.add_noise(action, action_noise, action_t)

        next_obs_noise = torch.randn_like(next_obs)
        next_obs_t = torch.randint(0, self.num_train_steps, (batch_size,), device=device).long()
        noisy_next_obs = self.noise_scheduler.add_noise(next_obs, next_obs_noise, next_obs_t)

        query_positions = generate_query_positions(
            num_points=self.num_points,
            device=device,
            method='grid'
        )
        query_positions = query_positions[None, :, :].expand(batch_size, -1, -1)

        (
            action_noise_pred,
            next_obs_noise_pred,
            coords_noise_pred,
            track_regression_pred_visibility,
        ) = self.noise_pred_net(
            obs,
            noisy_action,
            action_t,
            noisy_next_obs,
            next_obs_t,
            noisy_tracks_input,
            track_t,
            query_positions,
        )

        action_loss = F.mse_loss(action_noise_pred, action_noise)
        dynamics_loss = F.mse_loss(next_obs_noise_pred, next_obs_noise)
        coord_loss = F.mse_loss(coords_noise_pred, coords_noise)
        if self.use_track_visibility and track_vis_gt is not None:
            vis_loss = F.binary_cross_entropy_with_logits(
                track_regression_pred_visibility,
                track_vis_gt.float(),
            )
        else:
            vis_loss = track_regression_pred_visibility.sum() * 0.0

        loss = (
            action_loss
            + dynamics_loss
            + self.coord_loss_weight * coord_loss
            + self.vis_loss_weight * vis_loss
        )

        info = {
            "loss": loss.item(),
            "action_loss": action_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "coord_loss": coord_loss.item(),
            "vis_loss": vis_loss.item(),
        }

        if self.use_track_visibility and track_vis_gt is not None:
            with torch.no_grad():
                vis_pred_binary = (
                    torch.sigmoid(track_regression_pred_visibility) >= 0.5
                ).float()
                vis_accuracy = (vis_pred_binary == track_vis_gt).float().mean()
                info["vis_accuracy"] = vis_accuracy.item()

                true_pos = ((vis_pred_binary == 1) & (track_vis_gt == 1)).sum()
                pred_pos = (vis_pred_binary == 1).sum()
                actual_pos = (track_vis_gt == 1).sum()

                precision = true_pos / (pred_pos + 1e-8)
                recall = true_pos / (actual_pos + 1e-8)

                info["vis_precision"] = precision.item()
                info["vis_recall"] = recall.item()

        return loss, info

    def _visibility_probability(self, logits, *, force=False):
        if not force and not self.use_track_visibility:
            return None
        return torch.sigmoid(logits)

    @torch.no_grad()
    def sample_joint_with_visibility(self, obs_dict):
        """Sample actions, future observations, tracks, and visibility jointly."""
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)
        batch_size = obs_feat.shape[0]

        action_sample = torch.randn(
            (batch_size,) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (batch_size,) + self.latent_img_shape, device=obs_feat.device
        )

        track_coords_sample = torch.randn(
            batch_size, self.track_len, self.num_points, 2,
            device=obs_feat.device
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        query_positions = generate_query_positions(
            num_points=self.num_points,
            device=obs_feat.device,
            method='grid'
        )
        query_positions = query_positions[None, :, :].expand(batch_size, -1, -1)

        for t in self.noise_scheduler.timesteps:
            (
                action_noise_pred,
                next_obs_noise_pred,
                coords_noise_pred,
                vis_pred,
            ) = self.noise_pred_net(
                obs_feat,
                action_sample,
                t,
                next_obs_sample,
                t,
                track_coords_sample,
                t,
                query_positions,
            )

            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample

            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, t, next_obs_sample
            ).prev_sample

            track_coords_sample = self.noise_scheduler.step(
                coords_noise_pred, t, track_coords_sample
            ).prev_sample

        track_visibility = self._visibility_probability(vis_pred, force=True)

        return action_sample, next_obs_sample, track_coords_sample, track_visibility

    @torch.no_grad()
    def sample_joint(self, obs_dict):
        """Sample actions, future observations, tracks, and visibility jointly."""
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)
        batch_size = obs_feat.shape[0]

        action_sample = torch.randn(
            (batch_size,) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (batch_size,) + self.latent_img_shape, device=obs_feat.device
        )
        track_coords_sample = torch.randn(
            (batch_size,) + self.track_shape, device=obs_feat.device
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        query_positions = generate_query_positions(
            num_points=self.num_points,
            device=obs_feat.device,
            method='grid'
        )
        query_positions = query_positions[None, :, :].expand(batch_size, -1, -1)


        for t in self.noise_scheduler.timesteps:
            (
                action_noise_pred,
                next_obs_noise_pred,
                coords_noise_pred,
                vis_pred,
            ) = self.noise_pred_net(
                obs_feat,
                action_sample,
                t,
                next_obs_sample,
                t,
                track_coords_sample,
                t,
                query_positions,
            )

            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample

            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, t, next_obs_sample
            ).prev_sample

            track_coords_sample = self.noise_scheduler.step(
                coords_noise_pred, t, track_coords_sample
            ).prev_sample

        track_visibility = self._visibility_probability(vis_pred)

        return action_sample, next_obs_sample, track_coords_sample, track_visibility


    @torch.no_grad()
    def sample_forward_dynamics(self, obs_dict, action):
        """Sample future observations and tracks conditioned on actions."""
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)
        batch_size = obs_feat.shape[0]

        next_obs_sample = torch.randn(
            (batch_size,) + self.latent_img_shape, device=obs_feat.device
        )
        track_coords_sample = torch.randn(
            (batch_size,) + self.track_shape, device=obs_feat.device
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        action_t = torch.zeros(batch_size, dtype=torch.long, device=obs_feat.device)
        query_positions = generate_query_positions(
            num_points=self.num_points,
            device=obs_feat.device,
            method='grid'
        )
        query_positions = query_positions[None, :, :].expand(batch_size, -1, -1)
        for t in self.noise_scheduler.timesteps:
            (
                _,
                next_obs_noise_pred,
                coords_noise_pred,
                vis_pred,
            ) = self.noise_pred_net(
                obs_feat,
                action,
                action_t,
                next_obs_sample,
                t,
                track_coords_sample,
                t,
                query_positions,
            )

            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, t, next_obs_sample
            ).prev_sample

            track_coords_sample = self.noise_scheduler.step(
                coords_noise_pred, t, track_coords_sample
            ).prev_sample

        track_visibility = self._visibility_probability(vis_pred)
        return next_obs_sample, track_coords_sample, track_visibility


    @torch.no_grad()
    def sample_inverse_dynamics(self, obs_dict, next_obs_dict, tracks=None):
        """Sample actions conditioned on a future observation and optional tracks."""
        obs_feat, next_obs = self.obs_encoder.encode_curr_and_next_obs(
            obs_dict, next_obs_dict
        )
        batch_size = obs_feat.shape[0]

        action_sample = torch.randn(
            (batch_size,) + self.action_shape, device=obs_feat.device
        )

        if tracks is not None:
            if tracks.shape[-1] == 3:
                track_coords = tracks[..., :2]
            elif tracks.shape[-1] == 2:
                track_coords = tracks
            else:
                raise ValueError(f"Unexpected track shape: {tracks.shape}")

            track_sample = track_coords.to(obs_feat.device)
            track_t = torch.zeros(batch_size, dtype=torch.long, device=obs_feat.device)
        else:
            track_sample = torch.randn(
                (batch_size,) + self.track_shape, device=obs_feat.device
            )
            track_t = self.noise_scheduler.timesteps[0]

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        next_obs_t = torch.zeros(batch_size, dtype=torch.long, device=obs_feat.device)
        query_positions = generate_query_positions(
            num_points=self.num_points,
            device=obs_feat.device,
            method='grid'
        )
        query_positions = query_positions[None, :, :].expand(batch_size, -1, -1)
        for t in self.noise_scheduler.timesteps:
            (
                action_noise_pred,
                _,
                _,
                _,
            ) = self.noise_pred_net(
                obs_feat,
                action_sample,
                t,
                next_obs,
                next_obs_t,
                track_sample,
                track_t,
                query_positions,
            )

            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample

        return action_sample


    @torch.no_grad()
    def sample_marginal_next_obs(self, obs_dict):
        """Sample a future observation while marginalizing other modalities."""
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)
        batch_size = obs_feat.shape[0]

        action_sample = torch.randn(
            (batch_size,) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (batch_size,) + self.latent_img_shape, device=obs_feat.device
        )
        track_sample = torch.randn(
            (batch_size,) + self.track_shape, device=obs_feat.device
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        action_t = self.noise_scheduler.timesteps[0]
        track_t = self.noise_scheduler.timesteps[0]
        query_positions = generate_query_positions(
            num_points=self.num_points,
            device=obs_feat.device,
            method='grid'
        )
        query_positions = query_positions[None, :, :].expand(batch_size, -1, -1)
        for t in self.noise_scheduler.timesteps:
            (
                _,
                next_obs_noise_pred,
                _,
                _,
            ) = self.noise_pred_net(
                obs_feat,
                action_sample,
                action_t,
                next_obs_sample,
                t,
                track_sample,
                track_t,
                query_positions,
            )

            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, t, next_obs_sample
            ).prev_sample

        return next_obs_sample


    @torch.no_grad()
    def sample_marginal_tracks(self, obs_dict):
        """Sample tracks while marginalizing actions and future observations."""
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)
        batch_size = obs_feat.shape[0]

        action_sample = torch.randn(
            (batch_size,) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (batch_size,) + self.latent_img_shape, device=obs_feat.device
        )
        track_coords_sample = torch.randn(
            (batch_size,) + self.track_shape, device=obs_feat.device
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        action_t = self.noise_scheduler.timesteps[0]
        next_obs_t = self.noise_scheduler.timesteps[0]
        query_positions = generate_query_positions(
            num_points=self.num_points,
            device=obs_feat.device,
            method='grid'
        )
        query_positions = query_positions[None, :, :].expand(batch_size, -1, -1)
        for t in self.noise_scheduler.timesteps:
            (
                _,
                _,
                coords_noise_pred,
                vis_pred,
            ) = self.noise_pred_net(
                obs_feat,
                action_sample,
                action_t,
                next_obs_sample,
                next_obs_t,
                track_coords_sample,
                t,
                query_positions,
            )

            track_coords_sample = self.noise_scheduler.step(
                coords_noise_pred, t, track_coords_sample
            ).prev_sample

        track_visibility = self._visibility_probability(vis_pred)
        return track_coords_sample, track_visibility


    @torch.no_grad()
    def sample_with_next_obs(self, obs_dict, next_obs_dict):
        """Sample actions and tracks conditioned on a future observation."""
        obs_feat, next_obs = self.obs_encoder.encode_curr_and_next_obs(
            obs_dict, next_obs_dict
        )
        batch_size = obs_feat.shape[0]

        action_sample = torch.randn(
            (batch_size,) + self.action_shape, device=obs_feat.device
        )
        track_coords_sample = torch.randn(
            (batch_size,) + self.track_shape, device=obs_feat.device
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        next_obs_t = torch.zeros(batch_size, dtype=torch.long, device=obs_feat.device)
        query_positions = generate_query_positions(
            num_points=self.num_points,
            device=obs_feat.device,
            method='grid'
        )
        query_positions = query_positions[None, :, :].expand(batch_size, -1, -1)
        for t in self.noise_scheduler.timesteps:
            (
                action_noise_pred,
                _,
                coords_noise_pred,
                vis_pred,
            ) = self.noise_pred_net(
                obs_feat,
                action_sample,
                t,
                next_obs,
                next_obs_t,
                track_coords_sample,
                t,
                query_positions,
            )

            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample

            track_coords_sample = self.noise_scheduler.step(
                coords_noise_pred, t, track_coords_sample
            ).prev_sample

        track_visibility = self._visibility_probability(vis_pred)
        return action_sample, track_coords_sample, track_visibility


    @torch.no_grad()
    def sample_marginal_action(self, obs_dict):
        """Sample actions while marginalizing future observations and tracks."""
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)
        batch_size = obs_feat.shape[0]

        action_sample = torch.randn(
            (batch_size,) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (batch_size,) + self.latent_img_shape, device=obs_feat.device
        )
        track_sample = torch.randn(
            (batch_size,) + self.track_shape, device=obs_feat.device
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        next_obs_t = self.noise_scheduler.timesteps[0]
        track_t = self.noise_scheduler.timesteps[0]
        query_positions = generate_query_positions(
            num_points=self.num_points,
            device=obs_feat.device,
            method='grid'
        )
        query_positions = query_positions[None, :, :].expand(batch_size, -1, -1)
        for t in self.noise_scheduler.timesteps:
            action_noise_pred, _, _, _ = self.noise_pred_net(
                obs_feat,
                action_sample,
                t,
                next_obs_sample,
                next_obs_t,
                track_sample,
                track_t,
                query_positions,
            )

            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample

        return action_sample


    @torch.no_grad()
    def sample_action_and_tracks(self, obs_dict):
        """Sample actions and tracks while marginalizing future observations."""
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)
        batch_size = obs_feat.shape[0]

        action_sample = torch.randn(
            (batch_size,) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (batch_size,) + self.latent_img_shape, device=obs_feat.device
        )
        track_coords_sample = torch.randn(
            (batch_size,) + self.track_shape, device=obs_feat.device
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        next_obs_t = self.noise_scheduler.timesteps[0]
        query_positions = generate_query_positions(
            num_points=self.num_points,
            device=obs_feat.device,
            method='grid'
        )
        query_positions = query_positions[None, :, :].expand(batch_size, -1, -1)
        for t in self.noise_scheduler.timesteps:
            (
                action_noise_pred,
                _,
                coords_pred,
                vis_pred,
            ) = self.noise_pred_net(
                obs_feat,
                action_sample,
                t,
                next_obs_sample,
                next_obs_t,
                track_coords_sample,
                t,
                query_positions,
            )

            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample

            track_coords_sample = self.noise_scheduler.step(
                coords_pred, t, track_coords_sample
            ).prev_sample

        track_visibility = self._visibility_probability(vis_pred)
        return action_sample, track_coords_sample, track_visibility


    @torch.no_grad()
    def sample_action_with_tracks(self, obs_dict, tracks):
        """Sample actions conditioned on tracks."""
        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)
        batch_size = obs_feat.shape[0]

        action_sample = torch.randn(
            (batch_size,) + self.action_shape, device=obs_feat.device
        )

        next_obs_sample = torch.randn(
            (batch_size,) + self.latent_img_shape, device=obs_feat.device
        )

        if isinstance(tracks, torch.Tensor):
            if tracks.shape[-1] == 3:
                track_coords = tracks[..., :2]
            elif tracks.shape[-1] == 2:
                track_coords = tracks
            else:
                raise ValueError(f"Unexpected track shape: {tracks.shape}")

            track_sample = track_coords.to(obs_feat.device)
        else:
            track_sample = torch.tensor(tracks, device=obs_feat.device)
            if track_sample.shape[-1] == 3:
                track_sample = track_sample[..., :2]

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        next_obs_t = self.noise_scheduler.timesteps[0]

        track_t = torch.zeros(batch_size, dtype=torch.long, device=obs_feat.device)
        query_positions = generate_query_positions(
            num_points=self.num_points,
            device=obs_feat.device,
            method='grid'
        )
        query_positions = query_positions[None, :, :].expand(batch_size, -1, -1)
        for t in self.noise_scheduler.timesteps:
            action_noise_pred, _, _, _ = self.noise_pred_net(
                obs_feat,
                action_sample,
                t,
                next_obs_sample,
                next_obs_t,
                track_sample,
                track_t,
                query_positions,
            )

            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample

        return action_sample


    @torch.no_grad()
    def sample(self, obs_dict):
        """Sample marginal actions."""
        return self.sample_marginal_action(obs_dict)

    @torch.no_grad()
    def sample_next_obs_and_tracks(self, obs_dict, action):
        """Sample forward dynamics."""
        return self.sample_forward_dynamics(obs_dict, action)
