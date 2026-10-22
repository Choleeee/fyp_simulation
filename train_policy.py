#!/usr/bin/env python3
"""Train the approach policy by behaviour cloning.

    python3 train_policy.py

Reads simulation/data/*.npz written by iris_pole.py --collect or --dagger.
The saved action is the privileged expert command, including for DAgger files,
so rerunning this script after a DAgger rollout mixes the new labels in.
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from policy_net import ApproachPolicy
from task import ACTION_DIM, OBS_HEIGHT, OBS_WIDTH, STATE_DIM

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dataset(data_dir):
    paths = sorted(
        glob.glob(os.path.join(data_dir, "episode_*.npz"))
        + glob.glob(os.path.join(data_dir, "dagger_*.npz"))
    )
    if not paths:
        raise SystemExit(f"No episode files in {data_dir}. Run iris_pole.py --collect first.")
    depths, states, actions = [], [], []
    for path in paths:
        episode = np.load(path)
        depths.append(episode["depth"].astype(np.float32))
        states.append(episode["state"].astype(np.float32))
        actions.append(episode["action"].astype(np.float32))
    return np.concatenate(depths), np.concatenate(states), np.concatenate(actions), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(HERE, "data"))
    parser.add_argument("--out", default=os.path.join(HERE, "policy.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    depth, state, action, paths = load_dataset(args.data)
    print(f"{len(paths)} files, {len(action)} samples")
    if depth.shape[1:] != (OBS_HEIGHT, OBS_WIDTH) or state.shape[1] != STATE_DIM or action.shape[1] != ACTION_DIM:
        raise SystemExit(f"Unexpected sample shape: depth {depth.shape}, state {state.shape}, action {action.shape}")

    permutation = np.random.default_rng(0).permutation(len(action))
    split = max(1, int(0.1 * len(action)))
    val_index, train_index = permutation[:split], permutation[split:]

    def loader(index, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(depth[index, None, :, :]),
            torch.from_numpy(state[index]),
            torch.from_numpy(action[index]),
        )
        return DataLoader(dataset, batch_size=args.batch, shuffle=shuffle)

    train_loader = loader(train_index, True)
    val_loader = loader(val_index, False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ApproachPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        policy.train()
        train_loss = 0.0
        train_count = 0
        for depth_batch, state_batch, action_batch in train_loader:
            depth_batch = depth_batch.to(device)
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            loss = loss_fn(policy(depth_batch, state_batch), action_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss) * len(action_batch)
            train_count += len(action_batch)

        policy.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for depth_batch, state_batch, action_batch in val_loader:
                loss = loss_fn(
                    policy(depth_batch.to(device), state_batch.to(device)),
                    action_batch.to(device),
                )
                val_loss += float(loss) * len(action_batch)
                val_count += len(action_batch)
        print(
            f"epoch {epoch + 1:02d}  train {train_loss / train_count:.5f}  "
            f"val {val_loss / max(val_count, 1):.5f}"
        )

    torch.save(policy.state_dict(), args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
