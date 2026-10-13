"""
Graph Database Interface and Implementation for TRG Memory System

This module provides an abstraction layer for graph database operations,
with an in-memory NetworkX implementation that can be swapped for Neo4j.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import networkx as nx
from collections import deque

class NodeType(Enum):
    """Types of nodes in the memory graph"""
    EVENT = "EVENT"
    EPISODE = "EPISODE"  # Semantic grouping of related events
    NARRATIVE = "NARRATIVE"
    ENTITY = "ENTITY"
    SESSION = "SESSION"  # Session-level summary grouping

class LinkType(Enum):
    """Types of links between nodes"""
    TEMPORAL = "TEMPORAL"
    SEMANTIC = "SEMANTIC"
    CAUSAL = "CAUSAL"
    ENTITY = "ENTITY"  # Added as per design specification

class LinkSubType(Enum):
    """Subtypes for different link categories"""
    # Temporal subtypes
    PRECEDES = "PRECEDES"
    SUCCEEDS = "SUCCEEDS"
    CONCURRENT = "CONCURRENT"

    # Semantic subtypes
    RELATED_TO = "RELATED_TO"
    SIMILAR_TO = "SIMILAR_TO"
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"  # Episode contains events
    BELONGS_TO_SESSION = "BELONGS_TO_SESSION"  # Event belongs to session

    # Causal subtypes
    LEADS_TO = "LEADS_TO"
    BECAUSE_OF = "BECAUSE_OF"
    ENABLES = "ENABLES"
    PREVENTS = "PREVENTS"
    RESPONSE_TO = "RESPONSE_TO"  # For dialogue flow

    # Entity subtypes (as per design specification)
    REFERS_TO = "REFERS_TO"  # Event → Entity
    MENTIONED_IN = "MENTIONED_IN"  # Entity → Event

class LinkStatus(Enum):
    """Status of a link"""
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    PENDING = "PENDING"

@dataclass
class EventNode:
    """Represents an event node in the memory graph"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.EVENT
    timestamp: datetime = field(default_factory=datetime.now)
    content_narrative: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding_vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for storage"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "content_narrative": self.content_narrative,
            "attributes": self.attributes,
            "embedding_vector": self.embedding_vector.tolist() if hasattr(self.embedding_vector, "tolist") else self.embedding_vector
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventNode':
        """Create node from dictionary"""
        node = cls(
            node_id=data.get("node_id", str(uuid.uuid4())),
            content_narrative=data.get("content_narrative", ""),
            attributes=data.get("attributes", {}),
            embedding_vector=data.get("embedding_vector")
        )
        if "timestamp" in data and data["timestamp"]:
            node.timestamp = datetime.fromisoformat(data["timestamp"])
        if "node_type" in data:
            node.node_type = NodeType(data["node_type"])
        return node

@dataclass
class EpisodeNode:
    """Represents an episode node in the memory graph"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.EPISODE
    title: str = ""
    summary: str = ""
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    event_count: int = 0
    boundary_reason: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding_vector: Optional[List[float]] = None
    event_node_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for storage"""
        # Handle both datetime objects and strings for timestamps
        start_ts = None
        if self.start_timestamp:
            if isinstance(self.start_timestamp, datetime):
                start_ts = self.start_timestamp.isoformat()
            else:
                start_ts = str(self.start_timestamp)

        end_ts = None
        if self.end_timestamp:
            if isinstance(self.end_timestamp, datetime):
                end_ts = self.end_timestamp.isoformat()
            else:
                end_ts = str(self.end_timestamp)

        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "title": self.title,
            "summary": self.summary,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "event_count": self.event_count,
            "boundary_reason": self.boundary_reason,
            "attributes": self.attributes,
            "embedding_vector": self.embedding_vector.tolist() if hasattr(self.embedding_vector, "tolist") else self.embedding_vector,
            "event_node_ids": self.event_node_ids
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodeNode':
        """Create node from dictionary"""
        node = cls(
            node_id=data.get("node_id", str(uuid.uuid4())),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            event_count=data.get("event_count", 0),
            boundary_reason=data.get("boundary_reason", ""),
            attributes=data.get("attributes", {}),
            embedding_vector=data.get("embedding_vector"),
            event_node_ids=data.get("event_node_ids", [])
        )

        if data.get("start_timestamp"):
            node.start_timestamp = datetime.fromisoformat(data["start_timestamp"])
        if data.get("end_timestamp"):
            node.end_timestamp = datetime.fromisoformat(data["end_timestamp"])

        return node

@dataclass
class SessionNode:
    """Represents a session summary node in the memory graph"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.SESSION
    session_id: int = 0  # Session number (1, 2, 3, ...)
    summary: str = ""  # Full session summary text
    date_time: str = ""  # Session date/time
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding_vector: Optional[List[float]] = None
    event_node_ids: List[str] = field(default_factory=list)  # IDs of events in this session

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for storage"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "session_id": self.session_id,
            "summary": self.summary,
            "date_time": self.date_time,
            "attributes": self.attributes,
            "embedding_vector": self.embedding_vector.tolist() if hasattr(self.embedding_vector, "tolist") else self.embedding_vector,
            "event_node_ids": self.event_node_ids
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionNode':
        """Create node from dictionary"""
        node = cls(
            node_id=data.get("node_id", str(uuid.uuid4())),
            session_id=data.get("session_id", 0),
            summary=data.get("summary", ""),
            date_time=data.get("date_time", ""),
            attributes=data.get("attributes", {}),
            embedding_vector=data.get("embedding_vector"),
            event_node_ids=data.get("event_node_ids", [])
        )
        return node

@dataclass
class Link:
    """Represents a link/edge in the memory graph"""
    link_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = ""
    target_node_id: str = ""
    link_type: LinkType = LinkType.TEMPORAL
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize metadata with defaults"""
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now()
        if "status" not in self.metadata:
            self.metadata["status"] = LinkStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """Convert link to dictionary for storage"""
        return {
            "link_id": self.link_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "link_type": self.link_type.value,
            "properties": self.properties,
            "metadata": {
                **self.metadata,
                "created_at": self.metadata.get("created_at").isoformat()
                    if isinstance(self.metadata.get("created_at"), datetime) else self.metadata.get("created_at"),
                "status": self.metadata.get("status").value
                    if isinstance(self.metadata.get("status"), LinkStatus) else self.metadata.get("status")
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Link':
        """Create link from dictionary"""
        link = cls(
            link_id=data.get("link_id", str(uuid.uuid4())),
            source_node_id=data.get("source_node_id", ""),
            target_node_id=data.get("target_node_id", ""),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {})
        )
        if "link_type" in data:
            link.link_type = LinkType(data["link_type"])
        if "created_at" in link.metadata and isinstance(link.metadata["created_at"], str):
            link.metadata["created_at"] = datetime.fromisoformat(link.metadata["created_at"])
        if "status" in link.metadata and isinstance(link.metadata["status"], str):
            link.metadata["status"] = LinkStatus(link.metadata["status"])
        return link

@dataclass
class TraversalConstraints:
    """Constraints for graph traversal"""
    max_depth: int = 5
    max_nodes: int = 100
    link_types: Optional[Set[LinkType]] = None
    link_subtypes: Optional[Set[LinkSubType]] = None
    time_window: Optional[Tuple[datetime, datetime]] = None
    min_confidence: float = 0.0
    follow_temporal: bool = True
    follow_semantic: bool = True
    follow_causal: bool = True

    def allows_link(self, link: Link) -> bool:
        """Check if a link satisfies the constraints"""
        # Check link type
        if self.link_types and link.link_type not in self.link_types:
            return False

        # Check link subtype
        if self.link_subtypes:
            subtype = link.properties.get("sub_type")
            if subtype and LinkSubType(subtype) not in self.link_subtypes:
                return False

        # Check confidence score
        confidence = link.properties.get("confidence_score", 1.0)
        if confidence < self.min_confidence:
            return False

        # Check status
        if link.metadata.get("status") != LinkStatus.ACTIVE:
            return False

        # Check specific link type permissions
        if link.link_type == LinkType.TEMPORAL and not self.follow_temporal:
            return False
        if link.link_type == LinkType.SEMANTIC and not self.follow_semantic:
            return False
        if link.link_type == LinkType.CAUSAL and not self.follow_causal:
            return False

        return True

class GraphDBInterface(ABC):
    """Abstract interface for graph database operations"""

    @abstractmethod
    def add_node(self, node: EventNode) -> str:
        """Add a node to the graph"""
        pass

    @abstractmethod
    def add_link(self, link: Link) -> str:
        """Add a link between nodes"""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[EventNode]:
        """Retrieve a node by ID"""
        pass

    @abstractmethod
    def get_link(self, link_id: str) -> Optional[Link]:
        """Retrieve a link by ID"""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, link_type: Optional[LinkType] = None) -> List[Tuple[EventNode, Link]]:
        """Get neighboring nodes and their connecting links"""
        pass

    @abstractmethod
    def update_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """Update node properties"""
        pass

    @abstractmethod
    def update_link(self, link_id: str, updates: Dict[str, Any]) -> bool:
        """Update link properties"""
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its associated links"""
        pass

    @abstractmethod
    def delete_link(self, link_id: str) -> bool:
        """Delete a link"""
        pass

    @abstractmethod
    def traverse(self,
                start_nodes: List[str],
                constraints: TraversalConstraints) -> Dict[str, Any]:
        """Traverse the graph from starting nodes with constraints"""
        pass

    @abstractmethod
    def find_path(self, source_id: str, target_id: str,
                 link_type: Optional[LinkType] = None) -> Optional[List[str]]:
        """Find a path between two nodes"""
        pass

    @abstractmethod
    def get_subgraph(self, node_ids: List[str]) -> Dict[str, Any]:
        """Get a subgraph containing specified nodes"""
        pass

class NetworkXGraphDB(GraphDBInterface):
    """In-memory graph database implementation using NetworkX"""

    def __init__(self):
        """Initialize the NetworkX graph database"""
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, Union[EventNode, EpisodeNode]] = {}
        self.links: Dict[str, Link] = {}
        self.node_to_links: Dict[str, Set[str]] = {}

    def add_node(self, node: Union[EventNode, EpisodeNode]) -> str:
        """Add a node to the graph"""
        node_id = node.node_id
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())
        if node_id not in self.node_to_links:
            self.node_to_links[node_id] = set()
        return node_id

    def add_link(self, link: Link) -> str:
        """Add a link between nodes"""
        link_id = link.link_id
        self.links[link_id] = link

        if link.source_node_id not in self.graph:
            raise ValueError(f"Source node {link.source_node_id} not found")
        if link.target_node_id not in self.graph:
            raise ValueError(f"Target node {link.target_node_id} not found")

        self.graph.add_edge(
            link.source_node_id,
            link.target_node_id,
            key=link_id,
            **link.to_dict()
        )

        self.node_to_links.setdefault(link.source_node_id, set()).add(link_id)
        self.node_to_links.setdefault(link.target_node_id, set()).add(link_id)

        return link_id

    def get_node(self, node_id: str) -> Optional[Union[EventNode, EpisodeNode, SessionNode]]:
        """Retrieve a node by ID"""
        return self.nodes.get(node_id)

    def get_link(self, link_id: str) -> Optional[Link]:
        """Retrieve a link by ID"""
        return self.links.get(link_id)

    def get_neighbors(self, node_id: str, link_type: Optional[LinkType] = None) -> List[Tuple[Union[EventNode, EpisodeNode, SessionNode], Link]]:
        """Get neighboring nodes and their connecting links"""
        neighbors = []

        if node_id not in self.graph:
            return neighbors

        # Get outgoing edges
        for _, target_id, key, data in self.graph.out_edges(node_id, keys=True, data=True):
            link = self.links.get(key)
            if link and (link_type is None or link.link_type == link_type):
                target_node = self.nodes.get(target_id)
                if target_node:
                    neighbors.append((target_node, link))

        # Get incoming edges
        for source_id, _, key, data in self.graph.in_edges(node_id, keys=True, data=True):
            link = self.links.get(key)
            if link and (link_type is None or link.link_type == link_type):
                source_node = self.nodes.get(source_id)
                if source_node:
                    neighbors.append((source_node, link))

        return neighbors

    def update_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """Update node properties"""
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)

        self.graph.nodes[node_id].update(node.to_dict())
        return True

    def update_link(self, link_id: str, updates: Dict[str, Any]) -> bool:
        """Update link properties"""
        if link_id not in self.links:
            return False

        link = self.links[link_id]
        for key, value in updates.items():
            if key == "properties":
                link.properties.update(value)
            elif key == "metadata":
                link.metadata.update(value)
            elif hasattr(link, key):
                setattr(link, key, value)

        # Update graph edge attributes
        edge_data = self.graph.edges[link.source_node_id, link.target_node_id, link_id]
        edge_data.update(link.to_dict())
        return True

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its associated links"""
        if node_id not in self.nodes:
            return False

        links_to_delete = list(self.node_to_links.get(node_id, set()))
        for link_id in links_to_delete:
            self.delete_link(link_id)

        self.graph.remove_node(node_id)
        del self.nodes[node_id]
        if node_id in self.node_to_links:
            del self.node_to_links[node_id]

        return True

    def delete_link(self, link_id: str) -> bool:
        """Delete a link"""
        if link_id not in self.links:
            return False

        link = self.links[link_id]

        # Remove from graph
        try:
            self.graph.remove_edge(link.source_node_id, link.target_node_id, key=link_id)
        except:
            pass

        # Remove from tracking
        if link.source_node_id in self.node_to_links:
            self.node_to_links[link.source_node_id].discard(link_id)
        if link.target_node_id in self.node_to_links:
            self.node_to_links[link.target_node_id].discard(link_id)

        del self.links[link_id]
        return True

    def traverse(self,
                start_nodes: List[str],
                constraints: TraversalConstraints) -> Dict[str, Any]:
        """Traverse the graph from starting nodes with constraints"""
        visited_nodes = set()
        visited_links = set()
        traversal_paths = []
        node_depths = {}

        queue = deque([(node_id, 0, [node_id]) for node_id in start_nodes if node_id in self.nodes])

        while queue and len(visited_nodes) < constraints.max_nodes:
            current_id, depth, path = queue.popleft()

            if depth >= constraints.max_depth:
                continue

            if current_id in visited_nodes:
                continue

            visited_nodes.add(current_id)
            node_depths[current_id] = depth

            for neighbor_node, link in self.get_neighbors(current_id):
                if constraints.allows_link(link):
                    visited_links.add(link.link_id)

                    neighbor_id = neighbor_node.node_id
                    if neighbor_id not in visited_nodes:
                        new_path = path + [neighbor_id]
                        queue.append((neighbor_id, depth + 1, new_path))

                        if len(new_path) > 1:
                            traversal_paths.append(new_path)

        result_nodes = {node_id: self.nodes[node_id].to_dict()
                       for node_id in visited_nodes}
        result_links = {link_id: self.links[link_id].to_dict()
                       for link_id in visited_links}

        return {
            "nodes": result_nodes,
            "links": result_links,
            "paths": traversal_paths[:10],
            "node_depths": node_depths,
            "stats": {
                "nodes_visited": len(visited_nodes),
                "links_traversed": len(visited_links),
                "max_depth_reached": max(node_depths.values()) if node_depths else 0
            }
        }

    def find_path(self, source_id: str, target_id: str,
                 link_type: Optional[LinkType] = None) -> Optional[List[str]]:
        """Find a path between two nodes"""
        if source_id not in self.graph or target_id not in self.graph:
            return None

        try:
            # Treat the graph as undirected: the optimized build stores only
            # one direction per relation, so pathfinding must not depend on
            # the presence of reverse-copy edges.
            base = self.graph.to_undirected()
            if link_type:
                # Filter edges by link type
                def edge_filter(u, v, key):
                    link = self.links.get(key)
                    return link and link.link_type == link_type

                # Create subgraph with filtered edges
                subgraph = nx.subgraph_view(
                    base,
                    filter_edge=edge_filter
                )
                path = nx.shortest_path(subgraph, source_id, target_id)
            else:
                path = nx.shortest_path(base, source_id, target_id)

            return path
        except nx.NetworkXNoPath:
            return None

    def get_subgraph(self, node_ids: List[str]) -> Dict[str, Any]:
        """Get a subgraph containing specified nodes"""
        valid_node_ids = [nid for nid in node_ids if nid in self.nodes]

        if not valid_node_ids:
            return {"nodes": {}, "links": {}}

        subgraph = self.graph.subgraph(valid_node_ids)

        result_nodes = {nid: self.nodes[nid].to_dict() for nid in valid_node_ids}
        result_links = {}

        for u, v, key in subgraph.edges(keys=True):
            if key in self.links:
                result_links[key] = self.links[key].to_dict()

        return {
            "nodes": result_nodes,
            "links": result_links
        }

    def get_temporal_chain(self, start_node_id: str, direction: str = "forward",
                          max_hops: int = 10) -> List[EventNode]:
        """Get temporal chain of events.

        Compatible with both legacy graphs (TIME_PRECEDES + TIME_SUCCEEDS
        pairs) and the optimized single-direction build (TIME_PRECEDES
        only). Forward moves to later frames, backward to earlier frames,
        matching the direction semantics of KeyframeQueryEngine.
        """
        chain = []
        current_id = start_node_id
        visited = set()

        for _ in range(max_hops):
            if current_id in visited or current_id not in self.nodes:
                break

            visited.add(current_id)
            chain.append(self.nodes[current_id])

            # Find next temporal link
            next_id = None
            for neighbor, link in self.get_neighbors(current_id, LinkType.TEMPORAL):
                subtype = link.properties.get("sub_type")
                if direction == "forward":
                    # outgoing PRECEDES or incoming SUCCEEDS -> later frame
                    if (subtype == "TIME_PRECEDES"
                            and link.source_node_id == current_id):
                        next_id = link.target_node_id
                        break
                    if (subtype == "TIME_SUCCEEDS"
                            and link.target_node_id == current_id):
                        next_id = link.source_node_id
                        break
                elif direction == "backward":
                    # incoming PRECEDES or outgoing SUCCEEDS -> earlier frame
                    if (subtype == "TIME_PRECEDES"
                            and link.target_node_id == current_id):
                        next_id = link.source_node_id
                        break
                    if (subtype == "TIME_SUCCEEDS"
                            and link.source_node_id == current_id):
                        next_id = link.target_node_id
                        break

            if not next_id:
                break
            current_id = next_id

        return chain

    def find_causal_paths(self, source_id: str, max_depth: int = 3) -> List[List[str]]:
        """Find all causal paths from a source node"""
        paths = []

        def dfs_causal(current_id: str, path: List[str], depth: int):
            if depth >= max_depth:
                return

            for neighbor, link in self.get_neighbors(current_id, LinkType.CAUSAL):
                if link.properties.get("sub_type") in [LinkSubType.LEADS_TO.value, LinkSubType.ENABLES.value]:
                    if link.source_node_id == current_id:
                        next_id = link.target_node_id
                        if next_id not in path:
                            new_path = path + [next_id]
                            paths.append(new_path)
                            dfs_causal(next_id, new_path, depth + 1)

        if source_id in self.nodes:
            dfs_causal(source_id, [source_id], 0)

        return paths

    def export_to_json(self, filepath: str):
        """Export graph to JSON file"""
        export_data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "links": [link.to_dict() for link in self.links.values()]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)

    def import_from_json(self, filepath: str):
        """Import graph from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.graph.clear()
        self.nodes.clear()
        self.links.clear()
        self.node_to_links.clear()

        for node_data in data.get("nodes", []):
            node_type = node_data.get("node_type", "EVENT")
            if node_type == "EPISODE" or node_type == NodeType.EPISODE.value:
                node = EpisodeNode.from_dict(node_data)
            elif node_type == "SESSION" or node_type == NodeType.SESSION.value:
                node = SessionNode.from_dict(node_data)
            else:
                node = EventNode.from_dict(node_data)
            self.add_node(node)

        for link_data in data.get("links", []):
            link = Link.from_dict(link_data)
            self.add_link(link)

    def save(self, filepath: str):
        """Save the graph to a JSON file (alias for export_to_json)"""
        self.export_to_json(filepath)

    def load(self, filepath: str):
        """Load the graph from a JSON file (alias for import_from_json)"""
        self.import_from_json(filepath)
