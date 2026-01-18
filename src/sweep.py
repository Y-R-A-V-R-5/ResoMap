from pathlib import Path
import traceback
from tqdm import tqdm

class ExperimentSweep:
    """
    Experiment sweep orchestrator.
    
    Handles sequential execution of model training across multiple resolutions
    with support for selective execution and resume capability.
    """
    
    def __init__(self, runner, config, models_config, project_root):
        self.runner = runner
        self.config = config
        self.models_config = models_config
        self.project_root = project_root

    def run(self, dataset_path, models=None, resolutions=None, completed_runs=None):
        """
        Run experiments for selected models and resolutions.
        
        Parameters
        ----------
        dataset_path : str
            Path to dataset directory
        models : list, optional
            List of models to train. If None, uses all from config.
        resolutions : list, optional
            List of resolutions to train. If None, uses all from config.
        completed_runs : set, optional
            Set of tuples (model_name, resolution) that are already completed.
            Useful for skipping runs that succeeded in DagsHub.
        """
        models = models or self.config["sweep"]["models"]
        resolutions = resolutions or self.config["sweep"]["resolutions"]
        completed_runs = completed_runs or set()
        
        total_runs = len(models) * len(resolutions)
        completed_count = 0
        failed_count = 0
        
        print(f"\n{'='*60}")
        print(f"Sweep Configuration:")
        print(f"  Models: {len(models)} → {models}")
        print(f"  Resolutions: {len(resolutions)} → {resolutions}")
        print(f"  Total combinations: {total_runs}")
        print(f"  Already completed: {len(completed_runs)}")
        print(f"{'='*60}\n")

        for model_idx, model_name in enumerate(tqdm(models, desc="Models", position=0)):
            if model_name not in self.models_config:
                print(f"[Warning] Model '{model_name}' not found, skipping")
                continue

            for res_idx, res in enumerate(tqdm(
                resolutions, 
                desc=f"{model_name} (Res)", 
                leave=False,
                position=1
            )):
                # Check if already completed in DagsHub
                if (model_name, res) in completed_runs:
                    print(f"[Skip] {model_name}@{res} ✓ (completed in DagsHub)")
                    completed_count += 1
                    continue
                
                # Check if already has local checkpoint
                if self._has_checkpoint(model_name, res):
                    print(f"[Info] {model_name}@{res} has local checkpoint, may resume")

                print(f"\n{'='*60}")
                print(f"[{model_idx+1}/{len(models)}|{res_idx+1}/{len(resolutions)}] Training: {model_name} @ {res}x{res}")
                print(f"{'='*60}")

                try:
                    self.runner.run_experiment(
                        model_name=model_name,
                        resolution=res,
                        dataset_path=str(dataset_path),
                    )
                    completed_count += 1
                    print(f"[Success] {model_name}@{res} completed ✓")
                    
                except Exception as e:
                    failed_count += 1
                    print(f"\n[Error] {model_name}@{res} failed: {e}")
                    print(f"[Resume] To resume this model at this resolution, run:")
                    print(f"  python scripts/experiments.py --models {model_name} --resolutions {res}")
                    traceback.print_exc()
                    # Continue to next instead of breaking
                    continue

        # Final summary
        print(f"\n{'='*60}")
        print(f"Sweep Summary:")
        print(f"  Total combinations: {total_runs}")
        print(f"  Completed: {completed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Remaining: {total_runs - completed_count - failed_count}")
        print(f"{'='*60}")

    def _has_checkpoint(self, model_name, res):
        """
        Check if a checkpoint exists for the given model and resolution.
        
        This indicates a previous training attempt and allows resuming.
        """
        checkpoint_dir = (
            self.project_root / "checkpoints" / model_name / str(res)
        )
        return checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt"))
    
    def _already_done(self, model_name, res):
        """
        DEPRECATED: Use _has_checkpoint instead.
        
        Check if run directory exists locally.
        """
        run_dir = (
            self.project_root
            / "runs"
            / model_name
            / f"{res}x{res}"
        )
        return run_dir.exists()