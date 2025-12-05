# Goal

Implement a **standalone fact topic segmentation module** under the `mem0/` folder.

This module should:

- Take user facts as input.
- Route each fact into a **preset topic** and then into a **subtopic** inside that topic.
- Maintain a buffer per topic and periodically cluster buffered facts to create or extend subtopics.
- Use an embedding backend + LLM backend passed in from the outside (do **not** call OpenAI or any external API directly).
- Be testable **locally** on real data (no online services required; backends are pluggable).
- **Must NOT import or depend on any existing **``** code.** It should be a pure algorithm module that other parts of mem0 can call.

---

## Directory / Files

Create the following:

- `mem0/fact_topic_segmentation/__init__.py`
- `mem0/fact_topic_segmentation/segmenter.py`

The code must be completely self-contained, only using:

- Python standard library
- `numpy`
- `scikit-learn`

No imports from any other `mem0` modules.

---

## High-level design

We want a three-layer system:

1. **Topics** are preset (cold-start solved by manual topic definitions).
2. Each topic has **subtopics**, which are discovered and expanded over time.
3. A new fact goes through:
   - Topic routing (choose the best topic by cosine similarity to topic prototype embeddings, with a threshold).
   - Subtopic routing inside that topic:
     - If similar enough to an existing subtopic → assign directly.
     - Otherwise → push into that topic’s buffer for later clustering.
4. A batch process clusters buffered facts per topic and:
   - If a new cluster is similar enough to an existing subtopic → merge into that subtopic.
   - Otherwise → ask an LLM backend to decide whether:
     - It should reuse an existing subtopic, or
     - It should create a new subtopic name/description.

The module must be implemented as a reusable, pure-Python component that can be driven from tests or small local scripts.

---

## Public interfaces

Everything lives in `segmenter.py`.

Use `@dataclass` and type hints for all core structures.

### 1. `Fact`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Fact:
    id: str
    text: str
    user_id: Optional[str] = None
```

### 2. `Topic`

Represents a top-level topic (preset, cold-start seed).

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class Topic:
    id: str
    name: str
    description: str
    # embedding of (name + description)
    embedding: np.ndarray
```

### 3. `Subtopic`

Represents a subtopic inside a topic.

```python
from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class Subtopic:
    id: str
    topic_id: str
    name: str
    description: str
    embedding: np.ndarray    # prototype vector of this subtopic
    example_fact_ids: List[str] = field(default_factory=list)
```

### 4. `SubtopicLabelDecision`

This is what the LLM backend returns when we have a new cluster inside a topic.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SubtopicLabelDecision:
    kind: str  # "EXISTING" or "NEW"
    target_subtopic_id: Optional[str] = None
    new_subtopic_name: Optional[str] = None
    new_subtopic_description: Optional[str] = None
```

### 5. `FactAssignment`

The routing result for one fact.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class FactAssignment:
    fact_id: str
    topic_id: Optional[str]
    subtopic_id: Optional[str]
    finalized: bool = False  # False => in buffer & waiting for batch clustering
```

### 6. `TopicBuffer`

A per-topic buffer of facts that are not confidently assigned to any subtopic yet.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class TopicBuffer:
    topic_id: str
    fact_ids: List[str] = field(default_factory=list)
```

### 7. `TopicStore`

In-memory storage for topics, subtopics, and buffers.

```python
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TopicStore:
    topics: Dict[str, Topic] = field(default_factory=dict)
    subtopics: Dict[str, Dict[str, Subtopic]] = field(default_factory=dict)  # topic_id -> subtopic_id -> Subtopic
    buffers: Dict[str, TopicBuffer] = field(default_factory=dict)            # topic_id -> buffer

    def ensure_buffer(self, topic_id: str) -> TopicBuffer:
        """Return the buffer for this topic, creating it if needed."""
        ...

    def add_topic(self, topic: Topic) -> None:
        """Insert or replace a topic in the store."""
        ...

    def add_subtopic(self, subtopic: Subtopic) -> None:
        """Insert or replace a subtopic in the store."""
        ...
```

You must implement the bodies of `ensure_buffer`, `add_topic`, `add_subtopic`.

---

## Backends as protocols

We do not want hard dependencies on any specific embedding model or LLM.

Define two `Protocol` interfaces; the caller will inject concrete implementations.

### 8. `EmbeddingBackend`

```python
from typing import Protocol, List
import numpy as np

class EmbeddingBackend(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""
        ...
```

### 9. `LLMBackend`

```python
from typing import Protocol, List

class LLMBackend(Protocol):
    def decide_subtopic_label(
        self,
        topic_name: str,
        topic_description: str,
        candidate_subtopics: List[Subtopic],
        cluster_facts: List[Fact],
    ) -> SubtopicLabelDecision:
        """
        Given:
        - The parent topic name/description.
        - A list of existing subtopics under this topic.
        - A list of facts belonging to a new cluster.

        Decide:
        - Reuse an existing subtopic
          (kind="EXISTING", target_subtopic_id set), or
        - Create a new subtopic
          (kind="NEW", new_subtopic_name and optional description).
        """
        ...
```

Backend implementations for **real embeddings** and **real LLM calls** will be provided separately by the caller; this module must only depend on these Protocols.

For local testing you should also provide a minimal **dummy** implementation inside `if __name__ == "__main__":` that does something trivial (e.g. random embeddings + constant LLM decision), just to demonstrate end-to-end flow.

---

## Configuration

Define a simple config dataclass for thresholds and clustering parameters:

```python
from dataclasses import dataclass

@dataclass
class RoutingConfig:
    topic_threshold: float = 0.6              # min similarity to assign to a topic
    subtopic_assign_threshold: float = 0.7    # min similarity to assign directly to an existing subtopic
    subtopic_merge_threshold: float = 0.75    # min similarity to merge a new cluster into an existing subtopic
    buffer_min_size_for_clustering: int = 20  # min buffer size to trigger clustering
    k_min: int = 2
    k_max: int = 6
```

---

## Core class: `FactTopicSegmenter`

Implement a class `FactTopicSegmenter` that orchestrates everything.

```python
from typing import Optional, Tuple, List

class FactTopicSegmenter:
    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        llm_backend: LLMBackend,
        topic_store: TopicStore,
        config: Optional[RoutingConfig] = None,
    ) -> None:
        ...
```

### 1. Helper constructor for cold-start from preset topics

```python
    @classmethod
    def from_preset_topics(
        cls,
        embedding_backend: EmbeddingBackend,
        llm_backend: LLMBackend,
        preset_topics: List[Tuple[str, str, str]],  # (topic_id, name, description)
        config: Optional[RoutingConfig] = None,
    ) -> "FactTopicSegmenter":
        """
        Build a TopicStore from preset topics.
        For each topic, compute an embedding based on (name + description)
        using the embedding backend and save it into TopicStore.
        """
        ...
```

### 2. Single-fact online routing

```python
    def add_fact_and_route(self, fact: Fact) -> FactAssignment:
        """
        Main online entrypoint.

        Steps:
        1. Route to a topic by cosine similarity between fact embedding and topic embeddings.
           If best similarity < config.topic_threshold -> topic_id = None, finalized=False.
        2. If a topic is chosen:
           2.1 If the topic currently has no subtopics:
               - Push fact into that topic's buffer, subtopic_id=None, finalized=False.
           2.2 If the topic has subtopics:
               - Compute similarity to each subtopic embedding.
               - If max similarity >= config.subtopic_assign_threshold:
                     assign to that subtopic, update subtopic embedding
                     (e.g. moving average) and mark finalized=True.
               - Else:
                     push fact into the topic buffer, subtopic_id=None, finalized=False.
        """
        ...
```

### 3. Batch processing for buffers

```python
    def process_all_buffers(self) -> None:
        """
        For each topic buffer whose size >= buffer_min_size_for_clustering:
        - Embed all buffered facts.
        - Run clustering inside this topic.
        - For each cluster:
            * Compute cluster center (mean embedding).
            * Compare with existing subtopics via cosine similarity.
                - If max similarity >= subtopic_merge_threshold:
                    merge cluster into that subtopic (update embedding, add example ids).
                - Else:
                    call LLMBackend.decide_subtopic_label(...) to decide:
                        - reuse an existing subtopic, or
                        - create a new subtopic with a new name/description.
        - Clear processed buffers.
        """
        ...
```

### 4. Internal helpers

Implement these as private methods on `FactTopicSegmenter`:

- `_route_to_topic(text: str) -> Tuple[Optional[str], float]`
- `_route_within_topic(topic_id: str, fact: Fact) -> Tuple[Optional[str], bool]`
- `_update_subtopic_embedding(subtopic: Subtopic, new_vec: np.ndarray, alpha: float = 0.1) -> None`
- `_process_single_buffer(topic_id: str, buffer_: TopicBuffer) -> None`
- `_find_most_similar_subtopic(topic_id: str, cluster_center: np.ndarray) -> Optional[Tuple[Subtopic, float]]`
- `_call_llm_for_new_cluster(topic_id: str, cluster_facts: List[Fact], cluster_center: np.ndarray) -> SubtopicLabelDecision`
- `_generate_subtopic_id(topic_id: str) -> str` (simple deterministic id generation, e.g. `"{topic_id}_sub_{n+1}"`)

---

## Utilities

Implement simple cosine similarity and k selection using `MiniBatchKMeans`:

```python
import numpy as np
from typing import Tuple
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: shape (n, d)
    b: shape (m, d)
    return: shape (n, m)
    """
    ...


def select_kmeans_k(
    embs: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int = 42,
) -> Tuple[int, np.ndarray]:
    """
    Search k in [k_min, k_max] using silhouette_score and MiniBatchKMeans.
    Return (best_k, labels).
    If anything fails, fallback to k=2.
    """
    ...
```

### Algorithmic details

- Topic routing:

  - Use `EmbeddingBackend.embed([fact.text])`.
  - Compute cosine similarity with each `Topic.embedding`.
  - If best similarity < `topic_threshold`, return `topic_id=None`.

- Subtopic routing inside topic:

  - If no subtopics exist for this topic:
    - Push the fact into `TopicStore.buffers[topic_id]`.
  - If there are subtopics:
    - Compute cosine similarity with each `Subtopic.embedding`.
    - If max similarity >= `subtopic_assign_threshold`:
      - Assign to that subtopic, update its embedding via a simple moving average: `new_emb = (1-alpha) * old_emb + alpha * new_vec`.
    - Otherwise:
      - Push into buffer.

- Buffer clustering:

  - When a topic buffer size >= `buffer_min_size_for_clustering`, cluster all buffered facts for that topic.
  - For each cluster:
    - If cluster center similarity to some subtopic >= `subtopic_merge_threshold`, merge into that subtopic.
    - Else, call `LLMBackend.decide_subtopic_label` with:
      - parent topic name/description
      - all existing subtopics under that topic
      - the list of facts in this cluster
    - Use the returned decision to either:
      - reuse an existing subtopic id, or
      - create a new `Subtopic` with a new id, name, description, and prototype embedding = cluster center.

---

## Local testing / example usage

At the end of `segmenter.py`, add a small `if __name__ == "__main__":` block that:

- Creates a **dummy **`` (e.g. fixed small dimensional bag-of-words or random embeddings, but deterministic for the same text).
- Creates a **dummy **`` that:
  - Always returns `kind="NEW"` with a simple placeholder name like `"auto_subtopic"`.
- Initializes `FactTopicSegmenter.from_preset_topics(...)` with 2–3 preset topics.
- Feeds a few example facts and prints `FactAssignment` results.
- Calls `process_all_buffers()` and prints out the resulting topics/subtopics.

This is only for manual local sanity checking with real or synthetic data. There should be **no network calls** in this module.

---

## Important constraints

- Do **not** import from any other `mem0` modules.
- Only depend on `numpy` and `scikit-learn` plus the Python standard library.
- Code should be type-hinted and reasonably structured.
- Keep everything in `mem0/fact_topic_segmentation/segmenter.py` and `mem0/fact_topic_segmentation/__init__.py`.

You can assume other parts of the system (or local scripts) will:

- Construct a real `EmbeddingBackend` and `LLMBackend`.
- Call `FactTopicSegmenter` with real user facts.
- Persist `TopicStore` as needed outside this module.

