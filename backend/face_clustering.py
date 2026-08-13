"""Deterministic, quality-aware face clustering primitives."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .face_embeddings import normalize_embedding


def cosine(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(float(a) * float(b) for a, b in zip(left, right))


@dataclass(frozen=True)
class FaceSample:
    id: str
    embedding: list[float]
    quality: float
    pose_bucket: str = "unknown"
    model_name: str = "adaface"
    model_version: str = "unconfigured"
    protected_cluster: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "embedding", normalize_embedding(self.embedding))
        object.__setattr__(self, "quality", max(0.0, min(1.0, float(self.quality))))


@dataclass
class FacePrototype:
    sample_id: str
    embedding: list[float]
    quality: float
    pose_bucket: str


@dataclass
class FaceCluster:
    id: str
    members: list[FaceSample] = field(default_factory=list)
    prototypes: list[FacePrototype] = field(default_factory=list)
    noise: bool = False

    def best_score(self, sample):
        if not self.prototypes:
            return 0.0
        return max(cosine(sample.embedding, item.embedding) for item in self.prototypes)


@dataclass
class ClusterResult:
    clusters: list[FaceCluster]
    labels: dict[str, str]


class FaceClusterer:
    """Cluster views while avoiding low-quality bridge samples.

    High-quality view buckets may add a new prototype when they match an
    existing prototype. Unknown-view samples must also match every existing
    member, which prevents single-link bridge chaining.
    """

    def __init__(self, match_threshold=0.30, minimum_quality=0.30, prototype_limit=6):
        self.match_threshold = float(match_threshold)
        self.minimum_quality = float(minimum_quality)
        self.prototype_limit = int(prototype_limit)

    def fit(self, samples):
        clusters = []
        labels = {}
        ordered = sorted(samples, key=lambda item: (-item.quality, item.id))
        for sample in ordered:
            candidate = self._best_compatible_cluster(sample, clusters)
            if candidate is None:
                candidate = FaceCluster(
                    id=f"cluster-{len(clusters) + 1}",
                    noise=sample.quality < self.minimum_quality,
                )
                clusters.append(candidate)
            candidate.members.append(sample)
            self._update_prototypes(candidate, sample)
            labels[sample.id] = candidate.id
        return ClusterResult(clusters=clusters, labels=labels)

    def _best_compatible_cluster(self, sample, clusters):
        if sample.quality < self.minimum_quality:
            return None
        ranked = sorted(
            ((cluster.best_score(sample), cluster) for cluster in clusters if not cluster.noise),
            key=lambda item: (-item[0], item[1].id),
        )
        for score, cluster in ranked:
            if score < self.match_threshold:
                continue
            if any(item.model_name != sample.model_name or item.model_version != sample.model_version for item in cluster.members):
                continue
            protected = {item.protected_cluster for item in cluster.members if item.protected_cluster}
            if sample.protected_cluster and protected and sample.protected_cluster not in protected:
                continue
            if sample.pose_bucket == "unknown":
                if all(cosine(sample.embedding, member.embedding) >= self.match_threshold for member in cluster.members):
                    return cluster
            else:
                same_view = [member for member in cluster.members if member.pose_bucket == sample.pose_bucket]
                if not same_view or all(cosine(sample.embedding, member.embedding) >= self.match_threshold for member in same_view):
                    return cluster
        return None

    def _update_prototypes(self, cluster, sample):
        same_view = [item for item in cluster.prototypes if item.pose_bucket == sample.pose_bucket]
        if same_view:
            current = max(same_view, key=lambda item: item.quality)
            if sample.quality > current.quality:
                current.sample_id = sample.id
                current.embedding = sample.embedding
                current.quality = sample.quality
            return
        cluster.prototypes.append(
            FacePrototype(sample.id, sample.embedding, sample.quality, sample.pose_bucket)
        )
        cluster.prototypes.sort(key=lambda item: (-item.quality, item.pose_bucket, item.sample_id))
        del cluster.prototypes[self.prototype_limit :]


def pairwise_metrics(predicted, truth):
    """Return identity-pair precision/recall/F1 and merge diagnostics."""
    keys = sorted(set(predicted).intersection(truth))
    counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    for left, right in itertools.combinations(keys, 2):
        predicted_same = predicted[left] == predicted[right]
        truth_same = truth[left] == truth[right]
        if predicted_same and truth_same:
            counts["true_positive"] += 1
        elif predicted_same:
            counts["false_positive"] += 1
        elif truth_same:
            counts["false_negative"] += 1
        else:
            counts["true_negative"] += 1
    precision_denominator = counts["true_positive"] + counts["false_positive"]
    recall_denominator = counts["true_positive"] + counts["false_negative"]
    precision = counts["true_positive"] / precision_denominator if precision_denominator else 0.0
    recall = counts["true_positive"] / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    predicted_cluster_sizes = {}
    for label in predicted.values():
        predicted_cluster_sizes[label] = predicted_cluster_sizes.get(label, 0) + 1
    counts.update(
        precision=precision,
        recall=recall,
        f1=f1,
        singleton_ratio=(
            sum(size == 1 for size in predicted_cluster_sizes.values()) / len(predicted_cluster_sizes)
            if predicted_cluster_sizes
            else 0.0
        ),
    )
    return counts
