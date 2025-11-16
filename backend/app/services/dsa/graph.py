"""
Traffic graph representation for routing algorithms
"""
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq
from collections import defaultdict
import math


@dataclass
class Node:
    """Graph node representing an intersection"""
    id: int
    name: str
    latitude: float
    longitude: float
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id if isinstance(other, Node) else False


@dataclass
class Edge:
    """Graph edge representing a road segment"""
    source: int
    target: int
    segment_id: int
    length: float  # meters
    speed_limit: float  # km/h
    base_travel_time: float  # seconds
    congestion_factor: float = 1.0
    lanes: int = 1
    one_way: bool = False
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def current_travel_time(self) -> float:
        """Calculate current travel time based on congestion"""
        return self.base_travel_time * self.congestion_factor

    @property
    def weight(self) -> float:
        """Edge weight for routing algorithms"""
        return self.current_travel_time


class TrafficGraph:
    """
    Graph representation of road network for efficient routing

    Uses adjacency list representation for O(V + E) space complexity
    """

    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Dict[int, Edge]] = defaultdict(dict)  # adj list
        self._edge_by_segment: Dict[int, Edge] = {}

    def add_node(self, node: Node):
        """Add intersection node to graph"""
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        """
        Add road segment edge to graph

        Time complexity: O(1)
        """
        self.edges[edge.source][edge.target] = edge
        self._edge_by_segment[edge.segment_id] = edge

        # Add reverse edge if not one-way
        if not edge.one_way:
            reverse_edge = Edge(
                source=edge.target,
                target=edge.source,
                segment_id=edge.segment_id,
                length=edge.length,
                speed_limit=edge.speed_limit,
                base_travel_time=edge.base_travel_time,
                congestion_factor=edge.congestion_factor,
                lanes=edge.lanes,
                one_way=False,
                metadata=edge.metadata,
            )
            self.edges[edge.target][edge.source] = reverse_edge

    def get_neighbors(self, node_id: int) -> List[Tuple[int, Edge]]:
        """
        Get all neighbors of a node with edge information

        Time complexity: O(degree of node)
        """
        return [(target, edge) for target, edge in self.edges.get(node_id, {}).items()]

    def update_congestion(self, segment_id: int, congestion_factor: float):
        """
        Update congestion factor for a road segment

        Time complexity: O(1)
        """
        if segment_id in self._edge_by_segment:
            edge = self._edge_by_segment[segment_id]
            edge.congestion_factor = max(0.1, congestion_factor)

            # Update reverse edge if exists
            if not edge.one_way and edge.target in self.edges and edge.source in self.edges[edge.target]:
                self.edges[edge.target][edge.source].congestion_factor = edge.congestion_factor

    def get_node(self, node_id: int) -> Optional[Node]:
        """Get node by ID"""
        return self.nodes.get(node_id)

    def get_edge(self, source: int, target: int) -> Optional[Edge]:
        """Get edge between two nodes"""
        return self.edges.get(source, {}).get(target)

    def haversine_distance(self, node1_id: int, node2_id: int) -> float:
        """
        Calculate haversine distance between two nodes (heuristic for A*)

        Time complexity: O(1)
        """
        node1 = self.nodes.get(node1_id)
        node2 = self.nodes.get(node2_id)

        if not node1 or not node2:
            return float('inf')

        # Haversine formula
        R = 6371000  # Earth radius in meters
        lat1, lon1 = math.radians(node1.latitude), math.radians(node1.longitude)
        lat2, lon2 = math.radians(node2.latitude), math.radians(node2.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    def find_nearest_node(self, latitude: float, longitude: float) -> Optional[int]:
        """
        Find nearest intersection to given coordinates

        Time complexity: O(V) - can be optimized with spatial index
        """
        min_distance = float('inf')
        nearest_node_id = None

        for node_id, node in self.nodes.items():
            # Simple euclidean distance (could use haversine for accuracy)
            distance = math.sqrt(
                (node.latitude - latitude) ** 2 + (node.longitude - longitude) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                nearest_node_id = node_id

        return nearest_node_id

    def get_subgraph(self, node_ids: Set[int]) -> 'TrafficGraph':
        """Extract subgraph containing only specified nodes"""
        subgraph = TrafficGraph()

        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])

        for source in node_ids:
            for target, edge in self.edges.get(source, {}).items():
                if target in node_ids:
                    subgraph.add_edge(edge)

        return subgraph

    def get_stats(self) -> Dict:
        """Get graph statistics"""
        total_edges = sum(len(neighbors) for neighbors in self.edges.values())
        return {
            "num_nodes": len(self.nodes),
            "num_edges": total_edges,
            "avg_degree": total_edges / len(self.nodes) if self.nodes else 0,
        }

    def __len__(self) -> int:
        """Number of nodes in graph"""
        return len(self.nodes)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"TrafficGraph(nodes={stats['num_nodes']}, edges={stats['num_edges']})"
