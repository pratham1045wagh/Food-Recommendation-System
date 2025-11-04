import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Ensure pandas doesn't try to use pyarrow
pd.options.mode.copy_on_write = False
warnings.filterwarnings('ignore')

class DataVisualizer:
    """
    Handles data visualization for the food recommendation system.
    """
    
    def __init__(self, theme='dark'):
        self.theme = theme
        # Set style for matplotlib based on theme
        if theme == 'dark':
            plt.style.use('dark_background')
            self.mpl_bg_color = '#262730'
            self.mpl_text_color = '#FAFAFA'
            self.mpl_grid_color = '#444654'
        else:
            plt.style.use('default')
            sns.set_style("whitegrid")
            self.mpl_bg_color = '#FFFFFF'
            self.mpl_text_color = '#1F2937'  # Darker text for better contrast
            self.mpl_grid_color = '#E5E5E5'
        
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.facecolor'] = self.mpl_bg_color
        plt.rcParams['axes.facecolor'] = self.mpl_bg_color
        plt.rcParams['text.color'] = self.mpl_text_color
        plt.rcParams['axes.labelcolor'] = self.mpl_text_color
        plt.rcParams['axes.titlecolor'] = self.mpl_text_color
        plt.rcParams['xtick.color'] = self.mpl_text_color
        plt.rcParams['ytick.color'] = self.mpl_text_color
        plt.rcParams['axes.edgecolor'] = '#999999' if theme == 'light' else self.mpl_grid_color
        plt.rcParams['grid.color'] = '#E5E5E5' if theme == 'light' else self.mpl_grid_color
        plt.rcParams['grid.alpha'] = 0.8 if theme == 'light' else 0.5
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        if theme == 'light':
            plt.rcParams['axes.spines.left'] = True
            plt.rcParams['axes.spines.bottom'] = True
            plt.rcParams['axes.labelweight'] = 'bold'
            plt.rcParams['axes.titleweight'] = 'bold'
            plt.rcParams['xtick.labelsize'] = 10
            plt.rcParams['ytick.labelsize'] = 10
            plt.rcParams['axes.labelsize'] = 12
            plt.rcParams['axes.titlesize'] = 14
    
    def plot_top_items(self, data, top_n=10):
        """
        Plot top N best-selling items.
        
        Args:
            data: Order data DataFrame
            top_n: Number of top items to display
            
        Returns:
            plotly figure
        """
        # Calculate total quantity sold per item
        item_sales = data.groupby('Item Name')['Quantity'].sum().nlargest(top_n).sort_values()
        
        template = 'plotly_dark' if self.theme == 'dark' else 'none'
        colorscale = 'viridis' if self.theme == 'dark' else 'Blues'
        
        fig = px.bar(
            x=item_sales.values,
            y=item_sales.index,
            orientation='h',
            title=f'Top {top_n} Best-Selling Items',
            labels={'x': 'Quantity Sold', 'y': 'Item Name'},
            color=item_sales.values,
            color_continuous_scale=colorscale
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Quantity Sold",
            yaxis_title="Item Name",
            paper_bgcolor=self.mpl_bg_color,
            plot_bgcolor=self.mpl_bg_color,
            font_color=self.mpl_text_color,
            template=template,
            margin=dict(l=120, r=20, t=40, b=40),
            xaxis=dict(
                gridcolor=self.mpl_grid_color,
                linecolor=self.mpl_grid_color,
                title=dict(font=dict(color=self.mpl_text_color)),
                tickfont=dict(color=self.mpl_text_color)
            ),
            yaxis=dict(
                gridcolor=self.mpl_grid_color,
                linecolor=self.mpl_grid_color,
                title=dict(font=dict(color=self.mpl_text_color)),
                tickfont=dict(color=self.mpl_text_color, size=10)
            ),
            coloraxis_colorbar=dict(title_font_color=self.mpl_text_color, tickfont_color=self.mpl_text_color)
        )
        
        return fig
    
    def plot_revenue_by_item(self, data, top_n=10):
        """
        Plot revenue by item.
        
        Args:
            data: Order data DataFrame
            top_n: Number of top items to display
            
        Returns:
            plotly figure
        """
        # Calculate revenue per item
        item_revenue = data.groupby('Item Name')['Line_Total'].sum().nlargest(top_n).sort_values()
        
        template = 'plotly_dark' if self.theme == 'dark' else 'none'
        colorscale = 'viridis' if self.theme == 'dark' else 'Greens'
        
        fig = px.bar(
            x=item_revenue.values,
            y=item_revenue.index,
            orientation='h',
            title=f'Top {top_n} Items by Revenue',
            labels={'x': 'Revenue (£)', 'y': 'Item Name'},
            color=item_revenue.values,
            color_continuous_scale=colorscale
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Revenue (£)",
            yaxis_title="Item Name",
            paper_bgcolor=self.mpl_bg_color,
            plot_bgcolor=self.mpl_bg_color,
            font_color=self.mpl_text_color,
            template=template,
            margin=dict(l=120, r=20, t=40, b=40),
            xaxis=dict(
                gridcolor=self.mpl_grid_color,
                linecolor=self.mpl_grid_color,
                title=dict(font=dict(color=self.mpl_text_color)),
                tickfont=dict(color=self.mpl_text_color)
            ),
            yaxis=dict(
                gridcolor=self.mpl_grid_color,
                linecolor=self.mpl_grid_color,
                title=dict(font=dict(color=self.mpl_text_color)),
                tickfont=dict(color=self.mpl_text_color, size=10)
            ),
            coloraxis_colorbar=dict(title_font_color=self.mpl_text_color, tickfont_color=self.mpl_text_color)
        )
        
        return fig
    
    def plot_sales_trend(self, data):
        """
        Plot sales trend over time.
        
        Args:
            data: Order data DataFrame
            
        Returns:
            plotly figure
        """
        # Group by date
        daily_sales = data.groupby(data['Order Date'].dt.date).agg({
            'Order Number': 'nunique',
            'Line_Total': 'sum'
        }).reset_index()
        
        daily_sales.columns = ['Date', 'Orders', 'Revenue']
        
        template = 'plotly_dark' if self.theme == 'dark' else 'none'
        line_color1 = '#00A9E0' if self.theme == 'dark' else '#0066CC'
        line_color2 = '#21C354' if self.theme == 'dark' else '#008844'
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add orders trace
        fig.add_trace(
            go.Scatter(
                x=daily_sales['Date'],
                y=daily_sales['Orders'],
                name='Orders',
                mode='lines+markers',
                line=dict(color=line_color1, width=2),
                marker=dict(size=6)
            )
        )
        
        # Add revenue trace
        fig.add_trace(
            go.Scatter(
                x=daily_sales['Date'],
                y=daily_sales['Revenue'],
                name='Revenue (£)',
                mode='lines+markers',
                line=dict(color=line_color2, width=2),
                marker=dict(size=6),
                yaxis='y2'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Sales Trend Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Orders',
            yaxis2=dict(
                title=dict(text='Revenue (£)', font=dict(color=self.mpl_text_color)),
                overlaying='y',
                side='right',
                gridcolor=self.mpl_grid_color,
                tickfont=dict(color=self.mpl_text_color)
            ),
            hovermode='x unified',
            height=400,
            template=template,
            paper_bgcolor=self.mpl_bg_color,
            plot_bgcolor=self.mpl_bg_color,
            font_color=self.mpl_text_color,
            legend=dict(font_color=self.mpl_text_color),
            xaxis=dict(
                gridcolor=self.mpl_grid_color,
                linecolor=self.mpl_grid_color,
                title=dict(font=dict(color=self.mpl_text_color)),
                tickfont=dict(color=self.mpl_text_color)
            ),
            yaxis=dict(
                gridcolor=self.mpl_grid_color,
                linecolor=self.mpl_grid_color,
                title=dict(font=dict(color=self.mpl_text_color)),
                tickfont=dict(color=self.mpl_text_color)
            )
        )
        
        return fig
    
    def plot_cooccurrence_heatmap(self, data, top_n=15):
        """
        Plot item co-occurrence heatmap.
        
        Args:
            data: Order data DataFrame
            top_n: Number of top items to include
            
        Returns:
            matplotlib figure
        """
        # Get top N items
        top_items = data.groupby('Item Name')['Quantity'].sum().nlargest(top_n).index.tolist()
        
        # Filter data for top items
        filtered_data = data[data['Item Name'].isin(top_items)]
        
        # Create co-occurrence matrix
        cooccurrence = pd.DataFrame(0, index=top_items, columns=top_items)
        
        # Calculate co-occurrences
        for order_num in filtered_data['Order Number'].unique():
            order_items = filtered_data[filtered_data['Order Number'] == order_num]['Item Name'].tolist()
            
            for i, item1 in enumerate(order_items):
                for item2 in order_items[i:]:
                    if item1 in top_items and item2 in top_items:
                        cooccurrence.loc[item1, item2] += 1
                        if item1 != item2:
                            cooccurrence.loc[item2, item1] += 1
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10), facecolor=self.mpl_bg_color)
        ax.set_facecolor(self.mpl_bg_color)
        cmap = 'YlOrRd' if self.theme == 'dark' else 'YlGnBu'
        
        sns.heatmap(
            cooccurrence,
            annot=False,
            cmap=cmap,
            fmt='d',
            square=True,
            cbar_kws={'label': 'Co-occurrence Count'},
            ax=ax
        )
        
        # Set colorbar text color - ensure visibility in dark mode
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors=self.mpl_text_color, labelsize=10)
        cbar.ax.yaxis.label.set_color(self.mpl_text_color)
        cbar.ax.yaxis.label.set_fontsize(12)
        
        # Set axis text colors explicitly
        ax.set_xlabel('Items', fontsize=12, color=self.mpl_text_color, fontweight='bold')
        ax.set_ylabel('Items', fontsize=12, color=self.mpl_text_color, fontweight='bold')
        
        # Set tick labels colors
        ax.tick_params(axis='x', colors=self.mpl_text_color, labelsize=8)
        ax.tick_params(axis='y', colors=self.mpl_text_color, labelsize=8)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color=self.mpl_text_color)
        plt.setp(ax.get_yticklabels(), rotation=0, color=self.mpl_text_color)
        
        # Set title
        ax.set_title('Item Co-occurrence Heatmap', fontsize=14, fontweight='bold', 
                    color=self.mpl_text_color, pad=20)
        
        plt.tight_layout()
        
        return fig
    
    def plot_quantity_distribution(self, data):
        """
        Plot quantity distribution.
        
        Args:
            data: Order data DataFrame
            
        Returns:
            plotly figure
        """
        template = 'plotly_dark' if self.theme == 'dark' else 'none'
        color = '#00A9E0' if self.theme == 'dark' else '#0066CC'
        
        fig = px.histogram(
            data,
            x='Quantity',
            nbins=20,
            title='Item Quantity Distribution',
            labels={'Quantity': 'Quantity per Order', 'count': 'Frequency'},
            color_discrete_sequence=[color]
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Quantity per Order",
            yaxis_title="Frequency",
            paper_bgcolor=self.mpl_bg_color,
            plot_bgcolor=self.mpl_bg_color,
            font_color=self.mpl_text_color,
            template=template,
            xaxis=dict(
                gridcolor=self.mpl_grid_color,
                linecolor=self.mpl_grid_color,
                title=dict(font=dict(color=self.mpl_text_color)),
                tickfont=dict(color=self.mpl_text_color)
            ),
            yaxis=dict(
                gridcolor=self.mpl_grid_color,
                linecolor=self.mpl_grid_color,
                title=dict(font=dict(color=self.mpl_text_color)),
                tickfont=dict(color=self.mpl_text_color)
            )
        )
        
        return fig
    
    def plot_category_distribution(self, data, category_column='Item Name'):
        """
        Plot category distribution as pie chart.
        
        Args:
            data: Order data DataFrame
            category_column: Column to use for categories
            
        Returns:
            plotly figure
        """
        # Count items by category
        category_counts = data[category_column].value_counts().head(10)
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Top 10 Items Distribution',
            hole=0.3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        return fig
    
    def plot_hourly_heatmap(self, data):
        """
        Plot hourly order heatmap by day of week.
        
        Args:
            data: Order data DataFrame
            
        Returns:
            matplotlib figure
        """
        # Create pivot table
        hourly_data = data.groupby(['Day_Name', 'Hour'])['Order Number'].nunique().reset_index()
        hourly_pivot = hourly_data.pivot(index='Day_Name', columns='Hour', values='Order Number')
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hourly_pivot = hourly_pivot.reindex([day for day in day_order if day in hourly_pivot.index])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=self.mpl_bg_color)
        ax.set_facecolor(self.mpl_bg_color)
        cmap = 'YlOrRd' if self.theme == 'dark' else 'YlGnBu'
        
        sns.heatmap(
            hourly_pivot,
            annot=True,
            fmt='.0f',
            cmap=cmap,
            cbar_kws={'label': 'Number of Orders'},
            ax=ax,
            annot_kws={'color': self.mpl_text_color, 'fontsize': 10}
        )
        
        # Set colorbar text color - ensure visibility in dark mode
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors=self.mpl_text_color, labelsize=10)
        cbar.ax.yaxis.label.set_color(self.mpl_text_color)
        cbar.ax.yaxis.label.set_fontsize(12)
        
        # Set axis text colors explicitly
        ax.set_xlabel('Hour of Day', fontsize=12, color=self.mpl_text_color, fontweight='bold')
        ax.set_ylabel('Day of Week', fontsize=12, color=self.mpl_text_color, fontweight='bold')
        
        # Set tick labels colors
        ax.tick_params(axis='x', colors=self.mpl_text_color, labelsize=10)
        ax.tick_params(axis='y', colors=self.mpl_text_color, labelsize=10)
        
        # Set title
        ax.set_title('Order Heatmap by Day and Hour', fontsize=14, fontweight='bold', 
                    color=self.mpl_text_color, pad=20)
        
        plt.tight_layout()
        
        return fig
    
    def plot_order_value_distribution(self, data):
        """
        Plot order value distribution.
        
        Args:
            data: Order data DataFrame
            
        Returns:
            plotly figure
        """
        # Calculate order values
        order_values = data.groupby('Order Number')['Line_Total'].sum()
        
        fig = px.histogram(
            order_values,
            nbins=30,
            title='Order Value Distribution',
            labels={'value': 'Order Value (£)', 'count': 'Frequency'},
            color_discrete_sequence=['#00CC96']
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Order Value (£)",
            yaxis_title="Frequency"
        )
        
        return fig
    
    def plot_basket_size_distribution(self, data):
        """
        Plot basket size (items per order) distribution.
        
        Args:
            data: Order data DataFrame
            
        Returns:
            plotly figure
        """
        # Calculate basket sizes
        basket_sizes = data.groupby('Order Number')['Item Name'].count()
        
        fig = px.histogram(
            basket_sizes,
            nbins=20,
            title='Basket Size Distribution',
            labels={'value': 'Items per Order', 'count': 'Frequency'},
            color_discrete_sequence=['#AB63FA']
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Items per Order",
            yaxis_title="Frequency"
        )
        
        return fig

