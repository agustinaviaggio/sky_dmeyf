from ensemble_trainer import EnsembleTrainer
import logging

if __name__ == "__main__":
    trainer = EnsembleTrainer("ensemble_config.yaml")
    results = trainer.train_all()
    
    print("\n=== RESUMEN FINAL ===")
    print(results.group_by('study_name').agg([
        pl.count('semilla').alias('n_modelos'),
        pl.mean('ganancia_test1').alias('ganancia_test1_mean'),
        pl.mean('ganancia_test2').alias('ganancia_test2_mean')
    ]))