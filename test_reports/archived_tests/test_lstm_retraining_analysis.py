"""
LSTM Retraining Analysis & Optimal Configuration

Questions:
1. Should rf_retrain_increment/xgb_retrain_increment change with 100/60 setup?
2. Does LSTM need a separate retraining increment? (ANSWER: YES - currently hardcoded +5/+5)
3. What are optimal values for all models?
"""

import numpy as np


def analyze_increment_scaling():
    """
    Analyze if increments should scale with initial trial counts.
    """
    
    print("\n" + "="*90)
    print("🔬 RETRAINING INCREMENT ANALYSIS")
    print("="*90)
    
    print("\n📊 CURRENT SITUATION:")
    print("-"*90)
    
    configs = [
        {
            'model': 'RF',
            'old_trials': 500,
            'old_increment': 25,
            'new_trials': 100,
            'current_increment': 25
        },
        {
            'model': 'XGBoost',
            'old_trials': 300,
            'old_increment': 10,
            'new_trials': 60,
            'current_increment': 10
        },
        {
            'model': 'LSTM (trials)',
            'old_trials': 50,
            'old_increment': 5,
            'new_trials': 50,
            'current_increment': 5
        },
        {
            'model': 'LSTM (executions)',
            'old_trials': 10,
            'old_increment': 5,
            'new_trials': 10,
            'current_increment': 5
        }
    ]
    
    print(f"{'Model':<20} {'Old Setup':<20} {'Old Inc%':<12} {'New Setup':<20} {'New Inc%':<12} {'Status'}")
    print("-"*90)
    
    for config in configs:
        old_pct = (config['old_increment'] / config['old_trials']) * 100
        new_pct = (config['current_increment'] / config['new_trials']) * 100
        
        # Determine if increment needs adjustment
        if new_pct >= 20:
            status = "✅ Good"
        elif new_pct >= 10:
            status = "⚠️  Fair"
        else:
            status = "❌ Weak"
        
        print(f"{config['model']:<20} {config['old_trials']} + {config['old_increment']:<12} "
              f"{old_pct:>6.1f}%    {config['new_trials']} + {config['current_increment']:<12} "
              f"{new_pct:>6.1f}%    {status}")
    
    print("\n" + "="*90)
    print("🎯 KEY INSIGHT: Percentage Matters More Than Absolute Value")
    print("="*90)
    
    print("\n💡 RULE OF THUMB:")
    print("   • Excellent: >25% increment (fast convergence)")
    print("   • Good:      15-25% increment (steady progress)")
    print("   • Fair:      10-15% increment (slow progress)")
    print("   • Weak:      <10% increment (marginal improvement)")
    
    print("\n📊 ANALYSIS:")
    print("   • RF: 25% increment on 100 trials = GOOD ✅")
    print("   • XGBoost: 17% increment on 60 trials = GOOD ✅")
    print("   • LSTM trials: 10% increment on 50 trials = FAIR ⚠️")
    print("   • LSTM executions: 50% increment on 10 executions = EXCELLENT ✅")


def analyze_lstm_retraining():
    """
    Analyze LSTM retraining specifically.
    """
    
    print("\n" + "="*90)
    print("🚀 LSTM RETRAINING DEEP DIVE")
    print("="*90)
    
    print("\n❌ CURRENT PROBLEM:")
    print("   • LSTM has HARDCODED increments: +5 trials, +5 executions")
    print("   • No separate parameter like RF/XGBoost have")
    print("   • Located in ml_builder.py line ~1566:")
    print("     lstm_trials += 5")
    print("     lstm_executions += 5")
    
    print("\n📊 LSTM RETRAINING PROGRESSION:")
    print("-"*90)
    
    initial_trials = 50
    initial_execs = 10
    trial_inc = 5
    exec_inc = 5
    
    print(f"{'Attempt':<10} {'Trials':<12} {'Inc %':<12} {'Executions':<14} {'Inc %':<12} {'Total Evaluations'}")
    print("-"*90)
    
    for attempt in range(1, 6):
        trials = initial_trials + ((attempt - 1) * trial_inc)
        execs = initial_execs + ((attempt - 1) * exec_inc)
        total_evals = trials * execs
        
        if attempt == 1:
            trial_inc_pct = 0
            exec_inc_pct = 0
        else:
            prev_trials = initial_trials + ((attempt - 2) * trial_inc)
            prev_execs = initial_execs + ((attempt - 2) * exec_inc)
            trial_inc_pct = ((trials - prev_trials) / prev_trials) * 100
            exec_inc_pct = ((execs - prev_execs) / prev_execs) * 100
        
        print(f"Attempt {attempt:<3} {trials:<12} {trial_inc_pct:>6.1f}%    {execs:<14} "
              f"{exec_inc_pct:>6.1f}%    {total_evals:>8,}")
    
    print("\n⚠️  ISSUE: Trial increment is only 10% (weak for retraining)")


def recommend_optimal_config():
    """
    Recommend optimal configuration for all models.
    """
    
    print("\n" + "="*90)
    print("💡 RECOMMENDED CONFIGURATION")
    print("="*90)
    
    print("\n1️⃣  RF & XGBoost: KEEP CURRENT INCREMENTS")
    print("-"*90)
    print("   • RF: 100 trials, +25 increment = 25% growth ✅")
    print("   • XGBoost: 60 trials, +10 increment = 17% growth ✅")
    print("   • Rationale: Already in optimal range (15-25%)")
    
    print("\n2️⃣  LSTM: ADD SEPARATE PARAMETERS & ADJUST")
    print("-"*90)
    print("   • CURRENT: 50 trials + 5 = 10% (WEAK)")
    print("   • RECOMMENDED: 50 trials + 10 = 20% (GOOD)")
    print("   • RECOMMENDED: 10 executions + 2 = 20% (GOOD)")
    
    print("\n📝 IMPLEMENTATION:")
    print("-"*90)
    print("""
Function signature (add parameters):
    train_and_validate_models(
        ...,
        lstm_trials=50,
        lstm_executions=10,
        lstm_epochs=500,
        lstm_retrain_trials_increment=10,     # NEW
        lstm_retrain_executions_increment=2,  # NEW
        rf_trials=100,                         # UPDATED
        xgb_trials=60,                         # UPDATED
        rf_retrain_increment=25,
        xgb_retrain_increment=10,
        ...
    )

Retraining loop (replace hardcoded values):
    if not lstm_overfitted:
        print(f"✅ LSTM model accepted after {lstm_attempt + 1} attempt(s)")
        break
    elif lstm_attempt < max_retrains - 1:
        print(f"⚠️  Retraining LSTM with adjusted hyperparameters...")
        print(f"   Increasing trials: {lstm_trials} → {lstm_trials + lstm_retrain_trials_increment}")
        print(f"   Increasing executions: {lstm_executions} → {lstm_executions + lstm_retrain_executions_increment}")
        lstm_trials += lstm_retrain_trials_increment      # NEW PARAMETER
        lstm_executions += lstm_retrain_executions_increment  # NEW PARAMETER
    else:
        print(f"⚠️  LSTM reached maximum retrain attempts. Accepting current model.")
""")


def compare_configurations():
    """
    Compare different LSTM configurations.
    """
    
    print("\n" + "="*90)
    print("📊 LSTM CONFIGURATION COMPARISON")
    print("="*90)
    
    configs = [
        {
            'name': 'CURRENT (Hardcoded)',
            'trials': 50,
            'trial_inc': 5,
            'execs': 10,
            'exec_inc': 5
        },
        {
            'name': 'RECOMMENDED',
            'trials': 50,
            'trial_inc': 10,
            'execs': 10,
            'exec_inc': 2
        },
        {
            'name': 'AGGRESSIVE',
            'trials': 50,
            'trial_inc': 15,
            'execs': 10,
            'exec_inc': 3
        }
    ]
    
    print(f"\n{'Config':<25} {'Attempt 1':<15} {'Attempt 2':<15} {'Attempt 3':<15} {'Trial Inc %':<15} {'Exec Inc %'}")
    print("-"*90)
    
    for config in configs:
        a1_evals = config['trials'] * config['execs']
        a2_trials = config['trials'] + config['trial_inc']
        a2_execs = config['execs'] + config['exec_inc']
        a2_evals = a2_trials * a2_execs
        a3_trials = config['trials'] + (2 * config['trial_inc'])
        a3_execs = config['execs'] + (2 * config['exec_inc'])
        a3_evals = a3_trials * a3_execs
        
        trial_inc_pct = (config['trial_inc'] / config['trials']) * 100
        exec_inc_pct = (config['exec_inc'] / config['execs']) * 100
        
        print(f"{config['name']:<25} {a1_evals:>6,} evals   {a2_evals:>6,} evals   "
              f"{a3_evals:>6,} evals   {trial_inc_pct:>6.1f}%         {exec_inc_pct:>6.1f}%")
    
    print("\n✅ RECOMMENDED is optimal balance:")
    print("   • 20% trial growth (good for hyperparameter exploration)")
    print("   • 20% execution growth (good for stability validation)")
    print("   • Not too aggressive (avoids excessive training time)")


def test_improvements():
    """
    Test expected improvements with new configuration.
    """
    
    print("\n" + "="*90)
    print("📈 EXPECTED IMPROVEMENTS")
    print("="*90)
    
    print("\n🔧 CHANGES SUMMARY:")
    print("-"*90)
    print("1. RF trials: 50 → 100 (2x more exploration)")
    print("2. XGBoost trials: 30 → 60 (2x more exploration)")
    print("3. LSTM trial increment: +5 → +10 (2x faster growth)")
    print("4. LSTM execution increment: +5 → +2 (more stable growth)")
    print("5. All models: Separate retrain parameters (flexibility)")
    
    print("\n📊 EXPECTED OUTCOMES:")
    print("-"*90)
    print("✅ Better Initial Models:")
    print("   • 2x more hyperparameter exploration for RF/XGB")
    print("   • Higher chance of good initial model")
    
    print("\n✅ Fewer Retraining Attempts:")
    print("   • Better initial models = less retraining needed")
    print("   • Estimated: 3 attempts → 2 attempts (33% reduction)")
    
    print("\n✅ More Effective Retraining:")
    print("   • LSTM: 10% → 20% trial growth")
    print("   • Meaningful hyperparameter adjustments")
    
    print("\n✅ Parameter Flexibility:")
    print("   • Can tune increment per model independently")
    print("   • Can experiment with different strategies")
    
    print("\n⏱️  TIME IMPACT:")
    print("-"*90)
    print("First attempt: ~2x longer (but better quality)")
    print("Total time: Similar or faster (fewer retrains offset longer attempts)")


if __name__ == "__main__":
    analyze_increment_scaling()
    analyze_lstm_retraining()
    recommend_optimal_config()
    compare_configurations()
    test_improvements()
    
    print("\n" + "="*90)
    print("🎯 FINAL RECOMMENDATIONS")
    print("="*90)
    print("\n1. RF & XGBoost: KEEP increments (25 and 10) - already optimal")
    print("2. RF trials: 50 → 100 ✅")
    print("3. XGBoost trials: 30 → 60 ✅")
    print("4. LSTM: ADD separate increment parameters")
    print("5. LSTM trial increment: 5 → 10 (2x for better growth)")
    print("6. LSTM execution increment: 5 → 2 (more stable)")
    print("\n" + "="*90)
