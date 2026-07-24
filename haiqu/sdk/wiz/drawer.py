import plotly.graph_objects as go
import plotly.io as pio
from copy import deepcopy
import itertools
import numpy as np

# Set global style for scientific publication
pio.templates.default = "plotly_white"

# Color pairs: Gray, Blue, Red, Orange
COLORS_DARK = ["#333333", "#236CE6", "#FF455B", "#F27E3D"]  # Dark shades
COLORS_LIGHT = ["#D9D9D9", "#CDE9F7", "#F7CFC6", "#FFEABF"]  # Light shades (reordered to match)

LINESTYLES = ["solid", "dash", "dashdot", "dot", "longdash", "longdashdot"]
MARKERS = ["circle", "square", "diamond", "triangle-up", "triangle-down", "star", "pentagon", "hexagon", "cross", "x"]


class Drawer:
    """Enhanced drawer class for scientific publication-ready plots using Plotly."""

    def __init__(self, figsize=(900, 600), font_family="sans-serif", legend_font="sans-serif", font_size=14, grid=True):
        """Initialize the Drawer with customizable figure properties."""
        self.color_cycle = itertools.cycle(COLORS_DARK)
        self.linestyle_cycle = itertools.cycle(LINESTYLES)
        self.marker_cycle = itertools.cycle(MARKERS)
        self.fig = go.Figure()
        self.linestyle = LINESTYLES[0]
        self.marker = MARKERS[0]
        self.next_cycle_color = COLORS_DARK[0]
        self.color_counter = 0
        self.figsize = figsize
        self.font_family = font_family
        self.legend_font = legend_font
        self.font_size = font_size

        # Initialize figure with default layout
        self.fig.update_layout(
            width=figsize[0],
            height=figsize[1],
            font=dict(family=font_family, size=font_size),
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        # Set grid appearance
        if grid:
            self.fig.update_xaxes(
                showgrid=True,
                gridwidth=0.5,
                gridcolor="lightgray",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                tickfont=dict(family=font_family, size=font_size + 4),
            )
            self.fig.update_yaxes(
                showgrid=True,
                gridwidth=0.5,
                gridcolor="lightgray",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                tickfont=dict(family=font_family, size=font_size + 4),
            )
        self.last_call = None
        # Store the current active color to ensure consistency between plots and error bands
        self.current_active_color = None

    def _get_next_style(self):
        """Get the next color, line style, and marker in the cycle."""
        if self.color_counter == len(COLORS_DARK) - 1:
            self.color_counter = 0
            self.next_cycle_color = next(self.color_cycle)
            self.linestyle = next(self.linestyle_cycle)
        else:
            self.color_counter += 1
            self.next_cycle_color = next(self.color_cycle)

    def plot(
        self,
        x,
        y,
        label=None,
        color=None,
        linestyle=None,
        linewidth=2,
        marker=None,
        marker_size=8,
        show_markers=True,
        error_y=None,
        error_x=None,
        opacity=1.0,
        **kwargs,
    ):
        """Create a line plot with optional markers and error bars."""
        if self.last_call != "plot" and self.last_call != "scatter":
            self.clear()
        self.last_call = "plot"
        self._get_next_style()

        # flatten the list of lists
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        # Use provided color or take the next one from the cycle
        color = color or self.next_cycle_color
        # Store the actually used color for error bands to reference
        self.current_active_color = color

        linestyle = linestyle or self.linestyle
        # Always get a new marker
        self.marker = next(self.marker_cycle)
        marker_symbol = marker or self.marker

        mode = "lines" if not show_markers else "lines+markers"

        # Handle error bars
        error_y_dict = None
        if error_y is not None:
            if isinstance(error_y, dict):
                error_y_dict = error_y
            else:
                error_y_dict = dict(type="data", array=error_y, visible=True, color=color, thickness=linewidth / 2, width=4)

        error_x_dict = None
        if error_x is not None:
            if isinstance(error_x, dict):
                error_x_dict = error_x
            else:
                error_x_dict = dict(type="data", array=error_x, visible=True, color=color, thickness=linewidth / 2, width=4)

        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode=mode,
                line=dict(color=color, dash=linestyle, width=linewidth),
                marker=dict(symbol=marker_symbol, size=marker_size, color=color, line=dict(width=1, color="white")),
                opacity=opacity,
                name=label,
                error_y=error_y_dict,
                error_x=error_x_dict,
                **kwargs,
            )
        )
        self.fig.update_layout(width=self.figsize[0], height=self.figsize[1])

    def scatter(
        self,
        x,
        y,
        label=None,
        color=None,
        marker=None,
        marker_size=10,
        error_y=None,
        error_x=None,
        opacity=1.0,
        marker_line_width=1,
        marker_line_color="white",
        **kwargs,
    ):
        """Create a scatter plot with customizable markers and optional error bars."""
        if self.last_call != "plot" and self.last_call != "scatter":
            self.clear()
        self.last_call = "scatter"

        # flatten the list of lists
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        self._get_next_style()

        # Use provided color or take the next one from the cycle
        color = color or self.next_cycle_color
        # Store the actually used color for error bands to reference
        self.current_active_color = color

        # Always get a new marker
        self.marker = next(self.marker_cycle)
        marker_symbol = marker or self.marker

        # Handle error bars
        error_y_dict = None
        if error_y is not None:
            if isinstance(error_y, dict):
                error_y_dict = error_y
            else:
                error_y_dict = dict(type="data", array=error_y, visible=True, color=color, thickness=1, width=4)

        error_x_dict = None
        if error_x is not None:
            if isinstance(error_x, dict):
                error_x_dict = error_x
            else:
                error_x_dict = dict(type="data", array=error_x, visible=True, color=color, thickness=1, width=4)

        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    color=color,
                    symbol=marker_symbol,
                    size=marker_size,
                    opacity=opacity,
                    line=dict(width=marker_line_width, color=marker_line_color),
                ),
                opacity=opacity,
                name=label,
                error_y=error_y_dict,
                error_x=error_x_dict,
                **kwargs,
            )
        )
        self.fig.update_layout(width=self.figsize[0], height=self.figsize[1])

    def hist(self, data, bins=None, label=None, color=None, histnorm="", opacity=0.7, **kwargs):
        """Create a histogram with customizable properties."""
        if self.last_call != "hist":
            self.clear()
        self.last_call = "hist"
        self._get_next_style()
        # Use provided color or take the next one from the cycle
        color = color or self.next_cycle_color
        # Store the actually used color for error bands to reference
        self.current_active_color = color

        # Check if bins is an integer and raise error if not
        if bins is not None and not isinstance(bins, int):
            raise TypeError("The 'bins' parameter must be an integer")

        self.fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=bins,
                marker=dict(color=color, line=dict(width=1, color="white")),
                opacity=opacity,
                name=label,
                histnorm=histnorm,
                **kwargs,
            )
        )

        # Adjust layout for better histogram appearance
        self.fig.update_layout(bargap=0.1, width=self.figsize[0], height=self.figsize[1])

    def plot_histogram(
        self, counts_list, labels=None, title="Histograms", colors=None, opacity=0.7, barmode="group", bargap=0.1, **kwargs
    ):
        """Plot multiple histograms from dictionaries."""
        if self.last_call != "multiple_hists":
            self.clear()
        self.last_call = "multiple_hists"
        self.fig = go.Figure()

        if labels is None:
            labels = [f"Histogram {i + 1}" for i in range(len(counts_list))]
        elif len(labels) < len(counts_list):
            # Extend labels if there aren't enough
            labels = labels + [f"Histogram {i + 1}" for i in range(len(labels), len(counts_list))]

        colors = [] if colors is None else deepcopy(colors)

        # Make sure we have enough colors for all histograms
        while len(colors) < len(counts_list):
            colors.append(COLORS_DARK[len(colors) % len(COLORS_DARK)])

        num_entries = len(counts_list[0]) if counts_list else 1

        if num_entries > 10:
            dynamic_size = max(8, (self.font_size + 4) - (num_entries // 5))
        else:
            dynamic_size = self.font_size + 4

        # Add traces
        for i, counts in enumerate(counts_list):
            keys = list(counts.keys())
            values = list(counts.values())
            label = labels[i] if i < len(labels) else f"Histogram {i + 1}"

            self.fig.add_trace(
                go.Bar(
                    x=keys,
                    y=values,
                    marker_color=colors[i],
                    marker_line=dict(width=1, color="white"),
                    opacity=opacity,
                    name=label,
                    **kwargs,
                )
            )

        self.fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis=dict(
                type="category",
                tickmode="linear",
                dtick=1,
                tickangle=-45,  # Slanting is the best defense against overlap
                tickfont=dict(family=self.legend_font, size=dynamic_size),
            ),
            yaxis_title="Quasi-probabilities",
            width=self.figsize[0],
            height=self.figsize[1],
            barmode=barmode,
            bargap=bargap,
            showlegend=True,
            font=dict(family=self.font_family, size=self.font_size),
        )

    def set_title(self, title, size=None, color="black", x=0.5, y=0.9):
        """Set the title of the figure with enhanced formatting."""
        if size is None:
            size = self.font_size + 8

        self.fig.update_layout(title=dict(text=title, font=dict(family=self.font_family, size=size, color=color), x=x, y=y))

    def set_labels(self, xlabel=None, ylabel=None, font_size=None, color="black"):
        """Set the axis labels with enhanced formatting."""
        if font_size is None:
            font_size = self.font_size + 4

        if xlabel is not None:
            self.fig.update_xaxes(
                title=dict(text=xlabel, font=dict(family=self.font_family, size=font_size, color=color), standoff=15)
            )

        if ylabel is not None:
            self.fig.update_yaxes(
                title=dict(text=ylabel, font=dict(family=self.font_family, size=font_size, color=color), standoff=15)
            )

    def configure_legend(
        self,
        position="top-right",
        orientation="v",
        font_size=None,
        bgcolor="rgba(255,255,255,0.7)",
        border_color="gray",
        border_width=1,
    ):
        """Configure the legend with enhanced formatting.

        Parameters:
        -----------
        position : str
            Legend position. Options: 'top-right', 'top-left', 'bottom-right', 'bottom-left',
            'top', 'bottom'. Default: 'top-right'
        orientation : str
            Legend orientation: 'v' (vertical) or 'h' (horizontal). Default: 'v'
        font_size : int or None
            Font size for legend text. If None, uses the drawer's default font size.
        bgcolor : str
            Background color of the legend box. Default: 'rgba(255,255,255,0.7)' (semi-transparent white)
        border_color : str
            Color of the legend border. Default: 'gray'
        border_width : int
            Width of the legend border in pixels. Default: 1
        """
        if font_size is None:
            font_size = self.font_size

        # Validate orientation
        if orientation not in ["h", "v"]:
            raise ValueError(f"Unknown orientation: {orientation}. Use 'h' or 'v'")

        # Set legend position
        if position == "top-right":
            x, y, xanchor, yanchor = 1.0, 1.0, "right", "top"
        elif position == "top-left":
            x, y, xanchor, yanchor = 0.0, 1.0, "left", "top"
        elif position == "bottom-right":
            x, y, xanchor, yanchor = 1.0, 0.0, "right", "bottom"
        elif position == "bottom-left":
            x, y, xanchor, yanchor = 0.0, 0.0, "left", "bottom"
        elif position == "top":
            x, y, xanchor, yanchor = 0.5, 1.1, "center", "top"
        elif position == "bottom":
            x, y, xanchor, yanchor = 0.5, -0.2, "center", "bottom"
        else:
            raise ValueError(f"Unknown position: {position}")

        self.fig.update_layout(
            showlegend=True,
            legend=dict(
                font=dict(family=self.legend_font, size=font_size),
                orientation=orientation,
                x=x,
                y=y,
                xanchor=xanchor,
                yanchor=yanchor,
                bgcolor=bgcolor,
                bordercolor=border_color,
                borderwidth=border_width,
                tracegroupgap=5,
            ),
        )

    def add_horizontal_line(self, y, color="black", width=1.5, dash="dash", annotation=None):
        """Add a horizontal line to the figure."""
        self.fig.add_hline(y=y, line=dict(color=color, width=width, dash=dash))

        if annotation:
            self.fig.add_annotation(
                x=0.02, y=y, xref="paper", yref="y", text=annotation, showarrow=False, bgcolor="rgba(255,255,255,0.7)"
            )

    def add_vertical_line(self, x, color="black", width=1.5, dash="dash", annotation=None):
        """Add a vertical line to the figure."""
        self.fig.add_vline(x=x, line=dict(color=color, width=width, dash=dash))

        if annotation:
            self.fig.add_annotation(
                x=x, y=0.98, xref="x", yref="paper", text=annotation, showarrow=False, bgcolor="rgba(255,255,255,0.7)"
            )

    def add_error_band(self, x, y_lower, y_upper, color=None, opacity=0.2, name=None):
        """Add a shaded error band between two y-value arrays."""
        # If no color is specified, use the last used color instead of getting a new one
        if color is None:
            if self.current_active_color:
                # Use the same color as the most recent plot
                color = self.current_active_color
            else:
                # Only get a new color if we don't have a current active one
                self._get_next_style()
                color = self.next_cycle_color

        # Create a filled area for the error band
        x_band = list(x) + list(x[::-1])
        y_band = list(y_upper) + list(y_lower[::-1])

        self.fig.add_trace(
            go.Scatter(
                x=x_band,
                y=y_band,
                fill="toself",
                fillcolor=color,
                opacity=opacity,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=name is not None,
                name=name if name is not None else "Error Band",
            )
        )

    def add_annotation(
        self,
        text,
        x=None,
        y=None,
        xref="x",
        yref="y",
        showarrow=True,
        arrowhead=2,
        arrowcolor="black",
        font_size=None,
        bgcolor="rgba(255,255,255,0.7)",
    ):
        """
        Add annotation to the figure.

        Parameters:
        -----------
        text : str
            The text of the annotation.
        x : float or None
            x-coordinate of the annotation.
        y : float or None
            y-coordinate of the annotation.
        xref : str
            The x coordinate system ('x' for data coordinates, 'paper' for figure coordinates).
            Default: 'x' (data coordinates)
        yref : str
            The y coordinate system ('y' for data coordinates, 'paper' for figure coordinates).
            Default: 'y' (data coordinates)
        showarrow : bool
            Whether to show an arrow from the annotation to the point. Default: True
        arrowhead : int
            Arrow head style (0-8).
        arrowcolor : str
            Color of the arrow.
        font_size : int or None
            Font size for the annotation text. If None, uses self.font_size.
        bgcolor : str
            Background color for the annotation box.

        Note:
        -----
        - When using annotations with arrows (showarrow=True), 'x' and 'y' references
        are typically more intuitive as they refer to data coordinates.
        - For annotations without arrows or for positioning text relative to the figure,
        consider explicitly setting xref='paper' and yref='paper'.
        """
        if font_size is None:
            font_size = self.font_size

        self.fig.add_annotation(
            text=text,
            x=x,
            y=y,
            xref=xref,
            yref=yref,
            showarrow=showarrow,
            arrowhead=arrowhead,
            arrowcolor=arrowcolor,
            font=dict(family=self.font_family, size=font_size),
            bgcolor=bgcolor,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
        )

    def set_limits(self, xlim=None, ylim=None):
        if xlim:
            self.fig.update_xaxes(range=xlim)
        if ylim:
            self.fig.update_yaxes(range=ylim)

    def show(self):
        """Display the figure."""
        self.fig.show()

    def clear(self):
        self.fig = go.Figure()
        self.fig.update_layout(width=self.figsize[0], height=self.figsize[1])
        self.last_call = None
        self.current_active_color = None

    def close(self):
        """Close the figure (placeholder for compatibility)."""
        pass  # Plotly figures are not closed like matplotlib figures

    def save(self, filename, pathname=None):
        """Save the figure to a file.

        Args:
            filename: Name of the file (must end with .jpg, .jpeg, .pdf, or .png)
            pathname: Directory to save the file (default: current directory)
        """
        import os

        # Check file extension
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        # Verify extension is supported
        if ext not in [".jpg", ".jpeg", ".pdf", ".png"]:
            raise ValueError("File format not supported. Use .jpg, .jpeg, .pdf, or .png")

        # Use current directory if pathname is None
        if pathname is None:
            save_path = filename
        else:
            # Create directory if it doesn't exist
            os.makedirs(pathname, exist_ok=True)
            save_path = os.path.join(pathname, filename)

        # Save the figure
        self.fig.write_image(save_path, scale=2)
