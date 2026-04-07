"""
Migrate existing tuning directories to database and cleanup.

This script:
1. Scans tuning_dir for existing hyperparameter tuning folders
2. Extracts best hyperparameters from each folder
3. Saves them to the model_hyperparameters database table
4. Optionally deletes the folders after successful migration
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path


def extract_hyperparameters_from_folder(folder_path, model_type):
    """
    Extract best hyperparameters from a Keras Tuner or sklearn tuning folder.
    
    Parameters:
    - folder_path: Path to the tuning folder
    - model_type: Type of model (RF, XGB, LSTM, TCN)
    
    Returns:
    - dict with hyperparameters, or None if extraction failed
    """
    try:
        oracle_path = os.path.join(folder_path, "oracle.json")
        
        if not os.path.exists(oracle_path):
            # Check for sklearn tuner format (may have different structure)
            sklearn_oracle = os.path.join(folder_path, "trial_0", "oracle.json")
            if os.path.exists(sklearn_oracle):
                oracle_path = sklearn_oracle
            else:
                print(f"   [SKIP] No oracle.json found in {folder_path}")
                return None
        
        with open(oracle_path, 'r', encoding='utf-8') as f:
            oracle_data = json.load(f)
        
        # Extract best trial hyperparameters
        # Keras Tuner stores trials with their hyperparameters
        if 'hyperparameters' in oracle_data:
            # Direct hyperparameters in oracle
            return oracle_data['hyperparameters'].get('values', {})
        
        # Try to find best trial from ongoing_trials or end_order
        if 'ongoing_trials' in oracle_data or 'end_order' in oracle_data:
            # Look for trial files
            best_trial = None
            best_score = float('inf')
            
            trial_dirs = [d for d in os.listdir(folder_path) if d.startswith('trial_')]
            for trial_dir in trial_dirs:
                trial_path = os.path.join(folder_path, trial_dir, "trial.json")
                if os.path.exists(trial_path):
                    try:
                        with open(trial_path, 'r', encoding='utf-8') as f:
                            trial_data = json.load(f)
                        
                        # Get score (lower is better for loss/MAE)
                        score = trial_data.get('score', float('inf'))
                        if score is not None and score < best_score:
                            best_score = score
                            best_trial = trial_data
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            if best_trial and 'hyperparameters' in best_trial:
                hp_values = best_trial['hyperparameters'].get('values', {})
                return hp_values
        
        print(f"   [SKIP] Could not extract HP from {folder_path}")
        return None
        
    except Exception as e:
        print(f"   [ERROR] Failed to extract HP from {folder_path}: {e}")
        return None


def get_model_type_from_folder_name(folder_name):
    """Determine model type from folder name."""
    folder_lower = folder_name.lower()
    if 'rf_' in folder_lower or 'randomforest' in folder_lower or '_rf' in folder_lower:
        return 'RF'
    elif 'xgb' in folder_lower or 'xgboost' in folder_lower:
        return 'XGB'
    elif 'lstm' in folder_lower:
        return 'LSTM'
    elif 'tcn' in folder_lower:
        return 'TCN'
    else:
        return None


def get_ticker_from_folder_name(folder_name, model_type):
    """Extract ticker symbol from folder name."""
    # Common patterns:
    # RF_tuning_AAPL, XGB_tuning_GOOGL, LSTM_tuning_MSFT, TCN_tuning_TSLA
    # sklearn_RF_AAPL, etc.
    
    parts = folder_name.replace('-', '_').split('_')
    
    # Try to find the ticker (usually last part, all caps, 1-5 chars)
    for part in reversed(parts):
        if part.isupper() and 1 <= len(part) <= 6 and not part in ['RF', 'XGB', 'LSTM', 'TCN', 'TUNING', 'SKLEARN']:
            return part
    
    # Fallback: return the last part
    return parts[-1] if parts else folder_name


def migrate_tuning_directory(tuning_dir, dry_run=True, delete_after=False):
    """
    Migrate all tuning folders to database.
    
    Parameters:
    - tuning_dir: Path to tuning_dir
    - dry_run: If True, only show what would be done without saving
    - delete_after: If True, delete folders after successful migration
    
    Returns:
    - dict with migration statistics
    """
    from db_interactions import save_hyperparameters
    
    stats = {
        'total_folders': 0,
        'migrated': 0,
        'skipped': 0,
        'failed': 0,
        'deleted': 0,
        'space_freed_bytes': 0
    }
    
    if not os.path.exists(tuning_dir):
        print(f"Tuning directory not found: {tuning_dir}")
        return stats
    
    folders = [f for f in os.listdir(tuning_dir) if os.path.isdir(os.path.join(tuning_dir, f))]
    stats['total_folders'] = len(folders)
    
    print(f"\n{'='*60}")
    print(f"TUNING DIRECTORY MIGRATION")
    print(f"{'='*60}")
    print(f"Source: {tuning_dir}")
    print(f"Total folders: {len(folders)}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will save to DB)'}")
    print(f"Delete after: {'Yes' if delete_after else 'No'}")
    print(f"{'='*60}\n")
    
    for folder_name in sorted(folders):
        folder_path = os.path.join(tuning_dir, folder_name)
        
        # Calculate folder size
        folder_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(folder_path)
            for filename in filenames
        )
        folder_size_mb = folder_size / (1024 * 1024)
        
        model_type = get_model_type_from_folder_name(folder_name)
        if not model_type:
            print(f"[SKIP] {folder_name} - Unknown model type ({folder_size_mb:.1f} MB)")
            stats['skipped'] += 1
            continue
        
        ticker = get_ticker_from_folder_name(folder_name, model_type)
        
        print(f"\n[{model_type}] {folder_name}")
        print(f"   Ticker: {ticker}, Size: {folder_size_mb:.1f} MB")
        
        # Extract hyperparameters
        hp_dict = extract_hyperparameters_from_folder(folder_path, model_type)
        
        if hp_dict is None or len(hp_dict) == 0:
            print(f"   [SKIP] No hyperparameters extracted")
            stats['skipped'] += 1
            continue
        
        print(f"   Found {len(hp_dict)} hyperparameters")
        
        if dry_run:
            print(f"   [DRY RUN] Would save to database")
            stats['migrated'] += 1
        else:
            try:
                save_hyperparameters(
                    ticker=ticker,
                    model_type=model_type.lower(),  # Function expects lowercase
                    hyperparameters=hp_dict,
                    best_score=None,  # We don't have this in old format
                    num_trials=None
                )
                print(f"   [OK] Saved to database")
                stats['migrated'] += 1
                
                # Delete folder if requested
                if delete_after:
                    try:
                        shutil.rmtree(folder_path)
                        print(f"   [DELETED] Folder removed")
                        stats['deleted'] += 1
                        stats['space_freed_bytes'] += folder_size
                    except (OSError, PermissionError) as e:
                        print(f"   [WARN] Could not delete: {e}")
                        
            except Exception as e:
                print(f"   [ERROR] Failed to save: {e}")
                stats['failed'] += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MIGRATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total folders scanned: {stats['total_folders']}")
    print(f"Migrated to database:  {stats['migrated']}")
    print(f"Skipped (no HP found): {stats['skipped']}")
    print(f"Failed:                {stats['failed']}")
    if delete_after and not dry_run:
        print(f"Folders deleted:       {stats['deleted']}")
        print(f"Space freed:           {stats['space_freed_bytes'] / (1024**3):.2f} GB")
    print(f"{'='*60}\n")
    
    return stats


def main():
    """Main entry point with interactive prompts."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate tuning directories to database")
    parser.add_argument('--live', action='store_true', 
                        help='Actually save to database (default is dry run)')
    parser.add_argument('--delete', action='store_true', 
                        help='Delete folders after successful migration')
    parser.add_argument('--tuning-dir', type=str, default=None,
                        help='Path to tuning directory')
    
    args = parser.parse_args()
    
    # Default tuning directory
    if args.tuning_dir:
        tuning_dir = args.tuning_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tuning_dir = os.path.join(script_dir, "tuning_dir")
    
    dry_run = not args.live
    delete_after = args.delete
    
    if not dry_run:
        print("\n" + "="*60)
        print("WARNING: LIVE MODE - Changes will be saved to database!")
        if delete_after:
            print("WARNING: Folders will be DELETED after migration!")
        print("="*60)
        confirm = input("\nType 'yes' to continue: ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return
    
    stats = migrate_tuning_directory(
        tuning_dir=tuning_dir,
        dry_run=dry_run,
        delete_after=delete_after
    )
    
    if dry_run:
        print("\nThis was a DRY RUN. To actually migrate, run with --live flag.")
        print("To also delete folders after migration, add --delete flag.")
        print("\nExample: python migrate_tuning_to_db.py --live --delete")


if __name__ == "__main__":
    main()
