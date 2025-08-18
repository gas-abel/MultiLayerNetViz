

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.path import Path


plt.rcParams.update({'text.usetex': True})



def compute_user_colors(topic_alignement):
    # Number of users and topics
    NumU, P = topic_alignement.shape

    # Define a colormap (e.g., tab10 for categorical colors or viridis for gradients)
    colormap = plt.cm.tab10
    norm = Normalize(vmin=0, vmax=P - 1)

    # Create a mapping of topics to colors
    topic_colors = [colormap(norm(i))[:-1] for i in range(P)]

    # Compute user colors by blending topic colors based on alignment weights
    user_colors = np.zeros((NumU, 3))

    for u in range(NumU):
        user_colors[u] = np.sum(topic_colors * topic_alignement[u][:, None], axis=0)

    return user_colors, topic_colors

# Display the nodes with the right perspective
def create_parallelogram_marker(angle_deg=0):
    # Base parallelogram vertices (centered at origin)
    base_vertices = np.array([
        [-0.1, -0.5], # Bottom left
        [ 0.9, -0.5], # Bottom right
        [ 0.3,  0.5], # Top right
        [-0.7,  0.5], # Top left
        [-0.1, -0.5]  # Close the shape
    ])

    # Convert angle to radians and create rotation matrix
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # Rotate vertices
    rotated_vertices = base_vertices @ rotation_matrix.T

    # Create path
    return Path(rotated_vertices)

# Example usage
marker = create_parallelogram_marker(angle_deg=45)  # Rotate by 30°


def create_ellipse_marker(angle_deg=0, num_points=100):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = 0.5 * np.cos(t)
    y = 0.25 * np.sin(t)
    verts = np.column_stack((x, y))

    # Apply rotation
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    rotated_verts = verts @ R.T

    return Path(rotated_verts, closed=True)

# Usage
ellipse_marker = create_ellipse_marker(angle_deg=-5)


class LayeredNetworkGraph(object):

    def __init__(self, graphs, node_pairs=None, node_sizes=None, node_labels=None, node_color=None, layer_labels=None, edge_labels=None, node_markers='o', layouts=nx.spring_layout, ax=None):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Arguments:
        ----------
        graphs : list of networkx.Graph objects
            List of graphs, one for each layer.

        node_labels : dict node ID : str label or None (default None)
            Dictionary mapping nodes to labels.
            If None is provided, nodes are not labelled.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.

        """

        # book-keeping
        self.graphs = graphs
        self.total_layers = len(graphs)
        self.node_pairs = node_pairs
        self.node_sizes = node_sizes
        self.node_labels = node_labels
        self.node_color = node_color
        if len(node_markers)==1:
            self.node_markers = node_markers*self.total_layers
        else:
            self.node_markers = node_markers
        self.layout = layouts
        self.layer_labels = layer_labels
        self.edge_labels = edge_labels
        if ax:
            self.ax = ax
        else:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')

        # create internal representation of nodes and edges
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        # compute layout and plot
        self.get_node_positions()
        self.draw()


    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer)."""
        self.nodes = []
        for z, g in enumerate(self.graphs):
            self.nodes.extend([(node, z) for node in g.nodes()])


    def get_edges_within_layers(self):
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges()])


    def get_edges_between_layers(self):
        """Determine edges between layers based on a given set of node tuples.

        Arguments:
        ----------
        node_pairs : list of tuples
            List of tuples where each tuple contains two node IDs that should be connected between layers.
        """
        self.edges_between_layers = []
        node_pairs = self.node_pairs
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            if node_pairs is not None:
                for node1, node2 in node_pairs:
                    if node1 in g.nodes() and node2 in self.graphs[z2].nodes():
                        self.edges_between_layers.extend([((node1, z1), (node2, z2))])

    def get_node_positions(self,*args, **kwargs):
        """Get the node positions in the layered layout."""
        for z, g in enumerate(self.graphs):
            if z == 0:
                pos = self.layout[z](g, *args, **kwargs)
                self.node_positions = {(node, z) : (*pos[node], z) for node in g.nodes()}
            else:
                pos = self.layout[z](self.graphs[z], *args, **kwargs)
                self.node_positions.update({(node, z) : (*pos[node], z) for node in self.graphs[z].nodes()})


    def draw_nodes(self, nodes, color_nodes=None, size_nodes=None, marker_nodes='o', *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        if color_nodes is not None:
            colors = [color_nodes[node[0]] for node in nodes]
            if size_nodes is not None:
                sizes = [size_nodes[node[0]] for node in nodes]
                self.ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.8, marker=marker_nodes, *args, **kwargs)
            else:
                self.ax.scatter(x, y, z, c=colors, s=300, alpha=0.8, marker=marker_nodes, *args, **kwargs)
        else:
            if size_nodes is not None:
                sizes = [size_nodes[node[0]] for node in nodes]
                self.ax.scatter(x, y, z, s=sizes, alpha=0.8, marker=marker_nodes, *args, **kwargs)
            else:
                self.ax.scatter(x, y, z, s=300, alpha=0.8, marker=marker_nodes, *args, **kwargs)


    def draw_edges(self, edges, edge_labels=None, *args, **kwargs):
        segments = [(self.node_positions[source], self.node_positions[target]) for source, target in edges]
        line_collection = Line3DCollection(segments, *args, **kwargs)
        self.ax.add_collection3d(line_collection)
        for z in range(self.total_layers):
            if edge_labels:
                for (source, target), label in edge_labels[z].items():
                    pos_source = self.node_positions[(source,z)]
                    pos_target = self.node_positions[(target,z)]
                    pos_middle = [(3*pos_source[i] + pos_target[i]) / 4 for i in range(3)]
                    self.ax.text(*pos_middle, label)
                    if len(label) > 0:
                        # in that case draw a small arrow in the direction of the target
                        diff_pos = np.array(pos_target) - np.array(pos_source)
                        # Project onto XY-plane (set z to 0)
                        diff_pos_plane = np.array([diff_pos[0], diff_pos[1], 0.0])

                        # Normalize (to avoid length distortion)
                        norm = np.linalg.norm(diff_pos_plane)
                        if norm > 0:
                            diff_pos_plane /= norm

                        self.ax.quiver(*pos_middle, *diff_pos_plane, length=0.1, normalize=False,
                                    color='black', arrow_length_ratio=0.3)


    def get_extent(self, pad=0.1):
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = 1.1*np.min(xyz, axis=0)
        xmax, ymax, _ = 1.1*np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), \
            (ymin - pad * dy, ymax + pad * dy)


    def draw_plane(self, z, label=None, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u ,v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)
        if label:
            if len(label) == 2:
                [label_left,label_right] = label
                self.ax.text(0.9*xmin, 0.9*ymin, z, label_left, horizontalalignment='left', verticalalignment='bottom')
                self.ax.text(0.9*xmax, 0.9*ymax, z, label_right, horizontalalignment='right', verticalalignment='top')
            elif len(label) == 3:
                [label_left,label_right,label_mid] = label
                self.ax.text(0.9*xmin, 0.9*ymin, z, label_left, horizontalalignment='left', verticalalignment='bottom')
                self.ax.text(0.9*xmax, 0.9*ymax, z, label_right, horizontalalignment='right', verticalalignment='top')
                self.ax.text(0.9*xmax, 0.9*ymax, z/2, label_mid, horizontalalignment='right', verticalalignment='top', color='dimgrey')

    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z in self.nodes:
            if node in node_labels[z]:
                self.ax.text(*self.node_positions[(node, z)], node_labels[z][node], *args, **kwargs)


    def draw(self):
        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.1, zorder=2, label=self.layer_labels[z])

        self.draw_edges(self.edges_within_layers, edge_labels=self.edge_labels,  color='dimgrey', alpha=0.7, linestyle='-', zorder=1)
        if self.node_pairs is not None:
            self.draw_edges(self.edges_between_layers, color='dimgrey', alpha=0.3, linestyle='--', zorder=1)


        for z in range(self.total_layers):
            self.draw_nodes([node for node in self.nodes if node[1]==z], color_nodes = self.node_color[z], size_nodes=self.node_sizes[z], marker_nodes= self.node_markers[z], zorder=99)

        if self.node_labels:
            self.draw_node_labels(self.node_labels,
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  zorder=100)

