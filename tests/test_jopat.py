import unittest

import torch

from models.jopat import JOPAT


class DummyObservationEncoder(torch.nn.Module):
    def feat_dim(self):
        return 8

    def latent_img_shape(self):
        return (1, 4, 2, 4, 4)

    def encode_curr_obs(self, observations):
        batch_size = observations["dummy"].shape[0]
        return torch.zeros(
            batch_size,
            self.feat_dim(),
            device=observations["dummy"].device,
        )

    def encode_curr_and_next_obs(self, observations, next_observations):
        features = self.encode_curr_obs(observations)
        latents = torch.zeros(
            features.shape[0],
            *self.latent_img_shape(),
            device=features.device,
        )
        return features, latents


class JOPATTest(unittest.TestCase):
    def setUp(self):
        self.model = JOPAT(
            action_len=2,
            action_dim=3,
            track_len=2,
            num_points=25,
            obs_encoder=DummyObservationEncoder(),
            embed_dim=32,
            timestep_embed_dim=16,
            latent_patch_shape=(1, 2, 2),
            depth=1,
            num_heads=4,
            num_registers=2,
            num_train_steps=10,
            num_inference_steps=2,
            use_track_visibility=True,
        )

    def test_forward_and_backward(self):
        observations = {"dummy": torch.zeros(2, 1)}
        actions = torch.randn(2, 2, 3)
        coordinates = torch.rand(2, 2, 25, 2) * 2.0 - 1.0
        visibility = torch.randint(0, 2, (2, 2, 25, 1)).float()
        tracks = torch.cat((coordinates, visibility), dim=-1)

        loss, metrics = self.model(
            observations,
            observations,
            actions,
            tracks,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("vis_accuracy", metrics)
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        )

    def test_sampling_shapes_and_visibility_range(self):
        observations = {"dummy": torch.zeros(1, 1)}
        actions, tracks, visibility = self.model.sample_action_and_tracks(
            observations
        )

        self.assertEqual(tuple(actions.shape), (1, 2, 3))
        self.assertEqual(tuple(tracks.shape), (1, 2, 25, 2))
        self.assertEqual(tuple(visibility.shape), (1, 2, 25))
        self.assertTrue(torch.all((visibility >= 0.0) & (visibility <= 1.0)))


if __name__ == "__main__":
    unittest.main()
