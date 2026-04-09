from __future__ import annotations

from typing import List, Sequence

import torch

from agents.agent_memory import clear_cuda_cache


def _initialize_centroids_kmeans_pp(
    embeddings: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor.")
    num_points = embeddings.size(0)
    if num_points == 0:
        raise ValueError("embeddings must not be empty.")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive.")

    centroids = [embeddings[0]]
    if num_clusters == 1:
        return torch.stack(centroids, dim=0)

    min_distances = torch.cdist(embeddings, centroids[0].unsqueeze(0)).squeeze(1).pow(2)
    while len(centroids) < min(num_clusters, num_points):
        next_index = int(torch.argmax(min_distances).item())
        centroids.append(embeddings[next_index])
        distances = torch.cdist(embeddings, centroids[-1].unsqueeze(0)).squeeze(1).pow(2)
        min_distances = torch.minimum(min_distances, distances)
    return torch.stack(centroids, dim=0)


def _assign_clusters(
    embeddings: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    distances = torch.cdist(embeddings, centroids)
    return torch.argmin(distances, dim=1)


def _update_centroids(
    embeddings: torch.Tensor,
    assignments: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    updated = centroids.clone()
    for cluster_index in range(centroids.size(0)):
        members = embeddings[assignments == cluster_index]
        if members.numel() == 0:
            continue
        updated[cluster_index] = members.mean(dim=0)
    return updated


def select_centroid_token_indices(
    embeddings: torch.Tensor,
    num_clusters: int,
    num_iters: int = 20,
) -> List[int]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor.")
    num_points = embeddings.size(0)
    if num_points == 0:
        return []
    num_clusters = min(num_clusters, num_points)
    if num_clusters <= 0:
        return []
    if num_clusters == num_points:
        return list(range(num_points))

    work_embeddings = embeddings.detach().float().cpu()
    centroids = _initialize_centroids_kmeans_pp(work_embeddings, num_clusters)
    assignments = _assign_clusters(work_embeddings, centroids)
    for _ in range(max(1, num_iters)):
        updated_centroids = _update_centroids(work_embeddings, assignments, centroids)
        updated_assignments = _assign_clusters(work_embeddings, updated_centroids)
        if torch.equal(updated_assignments, assignments):
            centroids = updated_centroids
            assignments = updated_assignments
            break
        centroids = updated_centroids
        assignments = updated_assignments

    selected_indices: List[int] = []
    seen_indices: set[int] = set()
    for cluster_index in range(num_clusters):
        cluster_member_indices = torch.nonzero(assignments == cluster_index, as_tuple=False).flatten()
        if cluster_member_indices.numel() == 0:
            continue
        cluster_members = work_embeddings[cluster_member_indices]
        centroid = centroids[cluster_index].unsqueeze(0)
        closest_member_offset = int(torch.argmin(torch.cdist(cluster_members, centroid).squeeze(1)).item())
        selected_index = int(cluster_member_indices[closest_member_offset].item())
        if selected_index in seen_indices:
            continue
        seen_indices.add(selected_index)
        selected_indices.append(selected_index)

    if len(selected_indices) < num_clusters:
        for candidate_index in range(num_points):
            if candidate_index in seen_indices:
                continue
            selected_indices.append(candidate_index)
            seen_indices.add(candidate_index)
            if len(selected_indices) == num_clusters:
                break
    return selected_indices


def select_centroid_token_ids(
    *,
    token_ids: Sequence[int],
    embedding_weight: torch.Tensor,
    num_clusters: int,
    num_iters: int = 20,
) -> List[int]:
    if not token_ids:
        return []
    token_id_tensor = torch.tensor(list(token_ids), dtype=torch.long, device=embedding_weight.device)
    embeddings = embedding_weight.index_select(0, token_id_tensor)
    selected_indices = select_centroid_token_indices(
        embeddings=embeddings,
        num_clusters=num_clusters,
        num_iters=num_iters,
    )
    selected_token_ids = [int(token_ids[index]) for index in selected_indices]
    clear_cuda_cache()
    return selected_token_ids
