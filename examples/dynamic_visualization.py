"""
Dynamic real-time visualization for PHYSEC protocol demonstration.

This module provides a real-time dashboard that updates as the protocol runs,
showing current step, IQ data, spectrograms, and statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import threading
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)

class PhysecDynamicVisualizer:
    """Real-time visualization dashboard for PHYSEC protocol"""
    
    def __init__(self, max_runs=10):
        self.max_runs = max_runs
        self.current_run = 0
        
        # Protocol state tracking
        self.alice_step = "Idle"
        self.bob_step = "Idle"
        
        # Data storage for current run
        self.alice_iq_data = None
        self.bob_iq_data = None
        self.alice_spectrogram = None
        self.bob_spectrogram = None
        
        # Historical statistics
        self.bdr_history = deque(maxlen=max_runs)
        self.success_history = deque(maxlen=max_runs)
        self.run_numbers = deque(maxlen=max_runs)
        self.timing_history = deque(maxlen=max_runs)  # Key generation time in milliseconds
        
        # Colorbar tracking to prevent duplicates
        self.colorbars = {}
        
        # Threading and animation
        self.fig = None
        self.axes = {}
        self.animation = None
        self.running = False
        self.lock = threading.Lock()
        
        # Protocol step definitions
        self.protocol_steps = [
            "Idle",
            "Key Request", 
            "Probe TX",
            "Sample Collection",
            "PHYSEC Processing",
            "Parity Generation",
            "Reconciliation", 
            "Privacy Amplification",
            "Key Exchange",
            "Complete"
        ]
        
    def setup_visualization(self):
        """Initialize the matplotlib figure and subplots"""
        # Create figure with dark theme
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('PHYSEC Protocol - Real-time Visualization', 
                         fontsize=16, fontweight='bold', color='white')
        
        # Create grid layout (4 rows for more plots)
        gs = GridSpec(4, 4, figure=self.fig, hspace=0.4, wspace=0.3)
        
        # Step tracker (top row, spans 2 columns each)
        self.axes['alice_steps'] = self.fig.add_subplot(gs[0, 0:2])
        self.axes['bob_steps'] = self.fig.add_subplot(gs[0, 2:4])
        
        # IQ plots (second row)
        self.axes['alice_iq'] = self.fig.add_subplot(gs[1, 0:2])
        self.axes['bob_iq'] = self.fig.add_subplot(gs[1, 2:4])
        
        # Spectrograms (third row)
        self.axes['alice_spec'] = self.fig.add_subplot(gs[2, 0:2])
        self.axes['bob_spec'] = self.fig.add_subplot(gs[2, 2:4])
        
        # Statistics (bottom row)
        self.axes['bdr_stats'] = self.fig.add_subplot(gs[3, 0])
        self.axes['success_stats'] = self.fig.add_subplot(gs[3, 1])
        self.axes['timing_stats'] = self.fig.add_subplot(gs[3, 2:4])
        
        # Configure each subplot
        self._setup_step_trackers()
        self._setup_iq_plots()
        self._setup_spectrogram_plots()
        self._setup_statistics_plots()
        
        # Adjust layout with more space for colorbars and better spacing
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.85, top=0.92, wspace=0.4, hspace=0.4)
        
    def _setup_step_trackers(self):
        """Setup protocol step tracking visualizations"""
        for name, ax in [('alice_steps', self.axes['alice_steps']), 
                        ('bob_steps', self.axes['bob_steps'])]:
            node_name = name.split('_')[0].title()
            ax.set_title(f'{node_name} Protocol Steps', fontweight='bold')
            ax.set_xlim(-0.5, len(self.protocol_steps) - 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xticks(range(len(self.protocol_steps)))
            ax.set_xticklabels(self.protocol_steps, rotation=45, ha='right', fontsize=8)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
            
    def _setup_iq_plots(self):
        """Setup IQ data visualization"""
        for name, ax in [('alice_iq', self.axes['alice_iq']), 
                        ('bob_iq', self.axes['bob_iq'])]:
            node_name = name.split('_')[0].title()
            ax.set_title(f'{node_name} IQ Samples (Current Run)', fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 8192)
            ax.set_ylim(-2, 2)
            
    def _setup_spectrogram_plots(self):
        """Setup spectrogram visualization"""
        for name, ax in [('alice_spec', self.axes['alice_spec']), 
                        ('bob_spec', self.axes['bob_spec'])]:
            node_name = name.split('_')[0].title()
            ax.set_title(f'{node_name} Spectrogram', fontweight='bold', fontsize=10)
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=8)
            
    def _setup_statistics_plots(self):
        """Setup statistics tracking plots"""
        # BDR statistics
        ax = self.axes['bdr_stats']
        ax.set_title('Bit Disagreement Rate', fontweight='bold', fontsize=10)
        ax.set_xlabel('Run Number', fontsize=8)
        ax.set_ylabel('BDR (%)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.max_runs)
        ax.set_ylim(0, 30)
        
        # Success rate statistics
        ax = self.axes['success_stats']
        ax.set_title('Success Rate', fontweight='bold', fontsize=10)
        ax.set_xlabel('Run Number', fontsize=8)
        ax.set_ylabel('Success Rate (%)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.max_runs)
        ax.set_ylim(0, 100)
        
        # Key generation timing statistics
        ax = self.axes['timing_stats']
        ax.set_title('Key Generation Time', fontweight='bold', fontsize=10)
        ax.set_xlabel('Run Number', fontsize=8)
        ax.set_ylabel('Time (ms)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.max_runs)
        ax.set_ylim(0, 5000)  # Start with 5 second max, will auto-adjust
        
    def update_step(self, node, step):
        """Update the current protocol step for a node"""
        with self.lock:
            if node.lower() == 'alice':
                self.alice_step = step
            elif node.lower() == 'bob':
                self.bob_step = step
                
    def update_iq_data(self, node, iq_samples):
        """Update IQ sample data for a node"""
        with self.lock:
            if node.lower() == 'alice':
                self.alice_iq_data = iq_samples
                print(f"📊 Updated Alice IQ data: {type(iq_samples)}, shape: {iq_samples.shape if hasattr(iq_samples, 'shape') else 'N/A'}")
            elif node.lower() == 'bob':
                self.bob_iq_data = iq_samples
                print(f"📊 Updated Bob IQ data: {type(iq_samples)}, shape: {iq_samples.shape if hasattr(iq_samples, 'shape') else 'N/A'}")
                
    def update_spectrogram(self, node, spectrogram_data):
        """Update spectrogram data for a node"""
        with self.lock:
            if node.lower() == 'alice':
                self.alice_spectrogram = spectrogram_data
                print(f"📊 Updated Alice spectrogram: {type(spectrogram_data)}, shape: {spectrogram_data.shape if hasattr(spectrogram_data, 'shape') else 'N/A'}")
            elif node.lower() == 'bob':
                self.bob_spectrogram = spectrogram_data
                print(f"📊 Updated Bob spectrogram: {type(spectrogram_data)}, shape: {spectrogram_data.shape if hasattr(spectrogram_data, 'shape') else 'N/A'}")
                
    def add_run_statistics(self, bdr, success, timing_ms=None):
        """Add statistics for a completed run"""
        with self.lock:
            self.current_run += 1
            self.run_numbers.append(self.current_run)
            self.bdr_history.append(bdr * 100)  # Convert to percentage
            self.success_history.append(success)
            self.timing_history.append(timing_ms if timing_ms is not None else 0)
            
    def reset_for_new_run(self):
        """Reset data for a new protocol run"""
        with self.lock:
            self.alice_step = "Idle"
            self.bob_step = "Idle"
            # Keep IQ and spectrogram data until new data arrives
            # self.alice_iq_data = None
            # self.bob_iq_data = None
            # self.alice_spectrogram = None
            # self.bob_spectrogram = None
            
    def clear_data(self, node=None):
        """Clear data for a specific node or all nodes"""
        with self.lock:
            if node is None or node.lower() == 'alice':
                self.alice_iq_data = None
                self.alice_spectrogram = None
            if node is None or node.lower() == 'bob':
                self.bob_iq_data = None
                self.bob_spectrogram = None
            
    def _animate(self, frame):
        """Animation update function"""
        with self.lock:
            try:
                # Update step trackers
                self._update_step_display()
                
                # Update IQ plots
                self._update_iq_display()
                
                # Update spectrograms
                self._update_spectrogram_display()
                
                # Update statistics
                self._update_statistics_display()
                
            except Exception as e:
                logger.warning(f"Animation update error: {e}")
        
        return list(self.axes.values())
        
    def _update_step_display(self):
        """Update the protocol step display"""
        for name, step in [('alice_steps', self.alice_step), ('bob_steps', self.bob_step)]:
            ax = self.axes[name]
            ax.clear()
            
            node_name = name.split('_')[0].title()
            ax.set_title(f'{node_name} Protocol Steps - Current: {step}', fontweight='bold')
            ax.set_xlim(-0.5, len(self.protocol_steps) - 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xticks(range(len(self.protocol_steps)))
            ax.set_xticklabels(self.protocol_steps, rotation=45, ha='right', fontsize=8)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
            
            # Highlight current step
            if step in self.protocol_steps:
                step_idx = self.protocol_steps.index(step)
                ax.axvspan(step_idx - 0.4, step_idx + 0.4, alpha=0.5, color='green')
                
    def _update_iq_display(self):
        """Update IQ sample plots"""
        for name, data in [('alice_iq', self.alice_iq_data), ('bob_iq', self.bob_iq_data)]:
            ax = self.axes[name]
            ax.clear()
            
            node_name = name.split('_')[0].title()
            ax.set_title(f'{node_name} IQ Samples (Run #{self.current_run})', fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            
            if data is not None and len(data) > 0:
                # Plot I and Q components
                indices = np.arange(len(data))
                ax.plot(indices, np.real(data), 'b-', alpha=0.7, linewidth=0.5, label='I')
                ax.plot(indices, np.imag(data), 'r-', alpha=0.7, linewidth=0.5, label='Q')
                ax.legend(loc='upper right', fontsize=8)
                ax.set_xlim(0, len(data))
                
                # Calculate proper y-limits
                real_min, real_max = np.min(data.real), np.max(data.real)
                imag_min, imag_max = np.min(data.imag), np.max(data.imag)
                y_min = min(real_min, imag_min) * 1.1
                y_max = max(real_max, imag_max) * 1.1
                ax.set_ylim(y_min, y_max)
                
                # Add data info text
                info_text = f'Samples: {len(data)}\nMax I: {real_max:.3f}\nMax Q: {imag_max:.3f}'
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       ha='left', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            else:
                ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12, alpha=0.7, color='red')
                ax.set_xlim(0, 8192)
                ax.set_ylim(-2, 2)
                
    def _update_spectrogram_display(self):
        """Update spectrogram plots"""
        for name, data in [('alice_spec', self.alice_spectrogram), ('bob_spec', self.bob_spectrogram)]:
            ax = self.axes[name]
            
            # Remove existing colorbar if it exists
            if name in self.colorbars:
                try:
                    self.colorbars[name].remove()
                    del self.colorbars[name]
                except:
                    pass
            
            ax.clear()
            
            node_name = name.split('_')[0].title()
            ax.set_title(f'{node_name} Spectrogram', fontweight='bold', fontsize=10)
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=8)
            
            if data is not None and data.size > 0:
                # Display spectrogram
                im = ax.imshow(data, aspect='auto', origin='lower', 
                              cmap='viridis', interpolation='nearest')
                # Add colorbar and track it
                self.colorbars[name] = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
                
                # Add data info text
                info_text = f'Shape: {data.shape}\nMin: {np.min(data):.3f}\nMax: {np.max(data):.3f}'
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       ha='left', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=10, alpha=0.7, color='red')
                
    def _update_statistics_display(self):
        """Update statistics plots"""
        # BDR statistics
        ax = self.axes['bdr_stats']
        ax.clear()
        ax.set_title('Bit Disagreement Rate', fontweight='bold', fontsize=10)
        ax.set_xlabel('Run Number', fontsize=8)
        ax.set_ylabel('BDR (%)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(self.max_runs, self.current_run + 1))
        ax.set_ylim(0, 30)
        
        if len(self.bdr_history) > 0:
            ax.plot(list(self.run_numbers), list(self.bdr_history), 'o-', 
                   color='orange', linewidth=2, markersize=6)
            if len(self.bdr_history) > 1:
                avg_bdr = np.mean(self.bdr_history)
                ax.axhline(y=avg_bdr, color='red', linestyle='--', alpha=0.7, 
                          label=f'Avg: {avg_bdr:.1f}%')
                ax.legend(fontsize=8)
                
        # Success rate statistics
        ax = self.axes['success_stats']
        ax.clear()
        ax.set_title('Success Rate', fontweight='bold', fontsize=10)
        ax.set_xlabel('Run Number', fontsize=8)
        ax.set_ylabel('Success Rate (%)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(self.max_runs, self.current_run + 1))
        ax.set_ylim(0, 100)
        
        if len(self.success_history) > 0:
            # Calculate cumulative success rate first
            run_nums = list(self.run_numbers)
            cumulative_success = np.cumsum(self.success_history) / np.arange(1, len(self.success_history) + 1) * 100
            
            # Plot the trend line
            ax.plot(run_nums, cumulative_success, '-', 
                   color='blue', linewidth=2, alpha=0.7, label='Success Rate Trend')
            
            # Plot individual run results on the trend line
            for i, (run_num, success) in enumerate(zip(run_nums, self.success_history)):
                color = 'green' if success else 'red'
                marker = 'o' if success else 'X'
                # Place marker on the trend line at the corresponding success rate
                y_pos = cumulative_success[i]
                ax.scatter(run_num, y_pos, color=color, s=100, marker=marker, alpha=0.9, 
                          edgecolors='black', linewidth=1.5, zorder=5)
            
            if len(self.success_history) > 1:
                final_rate = cumulative_success[-1]
                ax.axhline(y=final_rate, color='blue', linestyle='--', alpha=0.5,
                          label=f'Current: {final_rate:.1f}%')
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                       markersize=8, label='Success', markeredgecolor='black'),
                Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
                       markersize=8, label='Failure', markeredgecolor='black'),
                Line2D([0], [0], color='blue', linewidth=2, alpha=0.7, label='Success Rate Trend')
            ]
            if len(self.success_history) > 1:
                legend_elements.append(Line2D([0], [0], color='blue', linestyle='--', 
                                            alpha=0.5, label=f'Current: {final_rate:.1f}%'))
            ax.legend(handles=legend_elements, fontsize=7, loc='lower right')
        
        # Key generation timing statistics
        ax = self.axes['timing_stats']
        ax.clear()
        ax.set_title('Key Generation Time', fontweight='bold', fontsize=10)
        ax.set_xlabel('Run Number', fontsize=8)
        ax.set_ylabel('Time (ms)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(self.max_runs, self.current_run + 1))
        
        if len(self.timing_history) > 0:
            # Auto-adjust y-axis based on data
            max_time = max(self.timing_history) if self.timing_history else 5000
            ax.set_ylim(0, max_time * 1.1)
            
            run_nums = list(self.run_numbers)
            timings = list(self.timing_history)
            
            # Plot individual run times
            ax.plot(run_nums, timings, 'o-', color='cyan', linewidth=2, markersize=6,
                   label='Individual Runs')
            
            # Plot running average
            if len(timings) > 1:
                avg_timing = np.mean(timings)
                ax.axhline(y=avg_timing, color='yellow', linestyle='--', alpha=0.7,
                          label=f'Average: {avg_timing:.0f} ms')
            
            # Add values on top of points for recent runs
            for i, (run_num, timing) in enumerate(zip(run_nums, timings)):
                if i >= len(run_nums) - 3:  # Show values for last 3 runs
                    ax.annotate(f'{timing:.0f}ms', 
                               (run_num, timing), 
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center', fontsize=7, color='white')
            
            ax.legend(fontsize=8, loc='upper right')
        else:
            ax.set_ylim(0, 5000)
                
    def start_visualization(self):
        """Start the real-time visualization"""
        if self.running:
            return
            
        self.setup_visualization()
        self.running = True
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self._animate, interval=500, blit=False, cache_frame_data=False
        )
        
        # Show in non-blocking mode
        plt.ion()
        plt.show()
        plt.draw()
        
        logger.info("Dynamic visualization started")
        
    def stop_visualization(self):
        """Stop the visualization"""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
        
        # Clean up colorbars
        for name, cbar in self.colorbars.items():
            try:
                cbar.remove()
            except:
                pass
        self.colorbars.clear()
        
        plt.ioff()
        plt.close(self.fig)
        logger.info("Dynamic visualization stopped")
        
    def update_display(self):
        """Update the display - call this from main thread periodically"""
        if self.running and self.fig:
            try:
                plt.figure(self.fig.number)
                plt.draw()
                plt.pause(0.001)  # Very short pause to process events
            except Exception as e:
                logger.warning(f"Display update error: {e}")
                
    def process_events(self):
        """Process matplotlib events - call this from main thread"""
        if self.running:
            try:
                plt.pause(0.01)  # Process GUI events
            except Exception as e:
                logger.warning(f"Event processing error: {e}")


# Global visualizer instance
_visualizer = None

def get_visualizer(max_runs=10):
    """Get or create the global visualizer instance"""
    global _visualizer
    if _visualizer is None:
        _visualizer = PhysecDynamicVisualizer(max_runs)
    return _visualizer

def start_dynamic_visualization(max_runs=10):
    """Start the dynamic visualization"""
    visualizer = get_visualizer(max_runs)
    visualizer.start_visualization()
    return visualizer

def stop_dynamic_visualization():
    """Stop the dynamic visualization"""
    global _visualizer
    if _visualizer:
        _visualizer.stop_visualization()
        _visualizer = None
