"""
Export utilities for Trust ABM
Save results to CSV, JSON, and pickle formats
"""

import json
import pickle
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime


class ResultsExporter:
    """
    Handle saving simulation results to various formats.
    
    Supports:
    - CSV (for easy analysis in Excel/R)
    - JSON (human-readable)
    - Pickle (full Python objects)
    """
    
    def __init__(self, output_dir='results'):
        """
        Initialize exporter.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save results (default 'results/')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Results will be saved to: {self.output_dir.absolute()}")
    
    def save_single_run(self, metrics, params, filename=None):
        """
        Save results from a single simulation run.
        
        Parameters
        ----------
        metrics : MetricsTracker
            Metrics from the run
        params : dict
            Parameters used
        filename : str, optional
            Filename (default: auto-generated with timestamp)
        
        Returns
        -------
        dict
            Paths to saved files
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"single_run_{timestamp}"
        
        # Create subdirectory for this run
        run_dir = self.output_dir / filename
        run_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # 1. Save time series to CSV
        csv_path = run_dir / "timeseries.csv"
        df = pd.DataFrame(metrics.history)
        df.to_csv(csv_path, index=False)
        saved_files['timeseries_csv'] = csv_path
        
        # 2. Save parameters to JSON
        params_path = run_dir / "parameters.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        saved_files['parameters'] = params_path
        
        # 3. Save summary to JSON
        summary_path = run_dir / "summary.json"
        summary = metrics.get_summary()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files['summary'] = summary_path
        
        # 4. Save full metrics object (pickle)
        pickle_path = run_dir / "metrics.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(metrics, f)
        saved_files['metrics_pickle'] = pickle_path
        
        print(f"\nSaved single run results to: {run_dir}")
        
        return saved_files
    
    def save_batch_runs(self, results, params, experiment_name=None):
        """
        Save results from multiple simulation runs.
        
        Parameters
        ----------
        results : list of MetricsTracker
            Results from all runs
        params : dict
            Parameters used (same for all runs)
        experiment_name : str, optional
            Name for this batch (default: auto-generated)
        
        Returns
        -------
        dict
            Paths to saved files
        """
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"batch_{timestamp}"
        
        # Create subdirectory
        batch_dir = self.output_dir / experiment_name
        batch_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # 1. Save aggregate summary to CSV
        summary_data = []
        for i, metrics in enumerate(results):
            summary = metrics.get_summary()
            summary['run_id'] = i
            summary_data.append(summary)
        
        df_summary = pd.DataFrame(summary_data)
        summary_path = batch_dir / "batch_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        saved_files['summary_csv'] = summary_path
        
        # 2. Save parameters
        params_path = batch_dir / "parameters.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        saved_files['parameters'] = params_path
        
        # 3. Save aggregate statistics
        from metrics import compute_share_trusting
        
        aggregate = {
            'n_runs': len(results),
            'share_trusting': compute_share_trusting(results),
            'avg_final_level_of_trust': np.mean([m.get_final_level_of_trust() for m in results]),
            'std_final_level_of_trust': np.std([m.get_final_level_of_trust() for m in results]),
            'n_converged_trust': sum(1 for m in results if m.converged_to_trust()),
            'n_converged_distrust': sum(1 for m in results if m.converged_to_distrust()),
        }
        
        aggregate_path = batch_dir / "aggregate_stats.json"
        with open(aggregate_path, 'w') as f:
            json.dump(aggregate, f, indent=2)
        saved_files['aggregate'] = aggregate_path
        
        # 4. Save all time series (for plotting)
        timeseries_dir = batch_dir / "timeseries"
        timeseries_dir.mkdir(exist_ok=True)
        
        for i, metrics in enumerate(results):
            ts_path = timeseries_dir / f"run_{i:03d}.csv"
            df = pd.DataFrame(metrics.history)
            df.to_csv(ts_path, index=False)
        
        saved_files['timeseries_dir'] = timeseries_dir
        
        # 5. Save full results (pickle)
        pickle_path = batch_dir / "all_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        saved_files['results_pickle'] = pickle_path
        
        print(f"\nSaved batch results to: {batch_dir}")
        print(f"  - {len(results)} runs")
        print(f"  - Share trusting: {aggregate['share_trusting']:.3f}")
        
        return saved_files
    
    def save_parameter_sweep(self, sweep_results, base_params, experiment_name=None):
        """
        Save results from a parameter sweep experiment.
        
        Handles both 1D and 2D sweeps.
        
        Parameters
        ----------
        sweep_results : dict
            Results from run_parameter_sweep() or run_2d_parameter_sweep()
        base_params : dict
            Base parameters
        experiment_name : str, optional
            Name for this experiment
        
        Returns
        -------
        dict
            Paths to saved files
        """
        # Detect if 1D or 2D sweep
        is_2d = 'param2_name' in sweep_results
        
        if experiment_name is None:
            if is_2d:
                param1_name = sweep_results['param1_name']
                param2_name = sweep_results['param2_name']
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_name = f"sweep_2d_{param1_name}_{param2_name}_{timestamp}"
            else:
                param_name = sweep_results['param_name']
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_name = f"sweep_{param_name}_{timestamp}"
        
        # Create subdirectory
        sweep_dir = self.output_dir / experiment_name
        sweep_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        if is_2d:
            # Handle 2D sweep
            param1_name = sweep_results['param1_name']
            param2_name = sweep_results['param2_name']
            param1_values = sweep_results['param1_values']
            param2_values = sweep_results['param2_values']
            share_trusting = sweep_results['share_trusting']
            avg_final_trust = sweep_results['avg_final_trust']
            
            # 1. Save summary CSV (one row per param2 value, columns are param1 values)
            summary_data = []
            for j, param2_val in enumerate(param2_values):
                row = {param2_name: param2_val}
                for i, param1_val in enumerate(param1_values):
                    row[f'{param1_name}_{param1_val}_share_trusting'] = share_trusting[j, i]
                    row[f'{param1_name}_{param1_val}_avg_final'] = avg_final_trust[j, i]
                summary_data.append(row)
            
            df_summary = pd.DataFrame(summary_data)
            summary_path = sweep_dir / "sweep_2d_summary.csv"
            df_summary.to_csv(summary_path, index=False)
            saved_files['summary_csv'] = summary_path
            
            # 2. Save in long format (easier for analysis)
            long_data = []
            for j, param2_val in enumerate(param2_values):
                for i, param1_val in enumerate(param1_values):
                    long_data.append({
                        param1_name: param1_val,
                        param2_name: param2_val,
                        'share_trusting': share_trusting[j, i],
                        'avg_final_trust': avg_final_trust[j, i]
                    })
            
            df_long = pd.DataFrame(long_data)
            long_path = sweep_dir / "sweep_2d_long_format.csv"
            df_long.to_csv(long_path, index=False)
            saved_files['long_csv'] = long_path
            
            # 3. Save metadata
            metadata = {
                'sweep_type': '2D',
                'param1_name': param1_name,
                'param1_values': param1_values,
                'param2_name': param2_name,
                'param2_values': param2_values,
                'n_runs_per_combination': len(sweep_results['all_results'][0][0]) if sweep_results['all_results'] else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        else:
            # Handle 1D sweep (original code)
            param_name = sweep_results['param_name']
            param_values = sweep_results['param_values']
            share_trusting = sweep_results['share_trusting']
            avg_final_trust = sweep_results['avg_final_trust']
            
            # 1. Save sweep summary to CSV
            df_sweep = pd.DataFrame({
                param_name: param_values,
                'share_trusting': share_trusting,
                'avg_final_trust': avg_final_trust
            })
            
            summary_path = sweep_dir / "sweep_summary.csv"
            df_sweep.to_csv(summary_path, index=False)
            saved_files['summary_csv'] = summary_path
            
            # 3. Save metadata
            metadata = {
                'sweep_type': '1D',
                'param_name': param_name,
                'param_values': param_values,
                'n_runs_per_value': len(sweep_results['all_results'][0]) if sweep_results['all_results'] else 0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Save base parameters (same for both)
        params_path = sweep_dir / "base_parameters.json"
        with open(params_path, 'w') as f:
            json.dump(base_params, f, indent=2)
        saved_files['parameters'] = params_path
        
        # Save metadata (same for both)
        metadata_path = sweep_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = metadata_path
        
        # Save full results (pickle) - same for both
        pickle_path = sweep_dir / "sweep_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(sweep_results, f)
        saved_files['results_pickle'] = pickle_path
        
        print(f"\nSaved parameter sweep to: {sweep_dir}")
        if is_2d:
            print(f"  - Parameter 1 (X-axis): {param1_name}")
            print(f"  - Parameter 2 (lines): {param2_name}")
            print(f"  - Combinations tested: {len(param1_values)} × {len(param2_values)}")
        else:
            print(f"  - Parameter: {param_name}")
            print(f"  - Values tested: {len(param_values)}")
        
        return saved_files
    def load_results(self, filepath):
        """
        Load results from pickle file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to pickle file
        
        Returns
        -------
        object
            Loaded results (MetricsTracker, list, or dict)
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        print(f"Loaded results from: {filepath}")
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def save_to_csv(data, filename, output_dir='results'):
    """
    Quick save dictionary or list to CSV.
    
    Parameters
    ----------
    data : dict or list of dict
        Data to save
    filename : str
        Output filename (will add .csv if missing)
    output_dir : str, optional
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not filename.endswith('.csv'):
        filename = filename + '.csv'
    
    filepath = output_dir / filename
    
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    
    df.to_csv(filepath, index=False)
    print(f"Saved to: {filepath}")
    
    return filepath


def save_to_json(data, filename, output_dir='results'):
    """
    Quick save dictionary to JSON.
    
    Parameters
    ----------
    data : dict
        Data to save
    filename : str
        Output filename (will add .json if missing)
    output_dir : str, optional
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not filename.endswith('.json'):
        filename = filename + '.json'
    
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved to: {filepath}")
    
    return filepath