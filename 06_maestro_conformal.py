class ConformalPredictor:
    """Implementa conformal prediction para clasificación binaria."""
    
    def __init__(self, models, feature_cols):
        self.models = models
        self.feature_cols = feature_cols
        self.calibration_scores = None
    
    def fit_calibration(self, X_cal, y_cal):
        """Calcula nonconformity scores en calibración."""
        # Promedio de probabilidades de todos los modelos
        all_probs = [model.predict(X_cal) for model in self.models]
        probs_ensemble = np.mean(all_probs, axis=0)
        
        # Nonconformity score: 1 - prob(clase_verdadera)
        self.calibration_scores = np.array([
            1 - probs_ensemble[i] if y_cal[i] == 1 else probs_ensemble[i]
            for i in range(len(y_cal))
        ])
        
        return self
    
    def calcular_confianza_individual(self, prob):
        """Calcula confianza (1-alpha) para una predicción individual."""
        score_0 = prob
        score_1 = 1 - prob
        
        quantile_0 = np.mean(self.calibration_scores >= score_0)
        quantile_1 = np.mean(self.calibration_scores >= score_1)
        
        alpha = max(quantile_0, quantile_1)
        confidence = 1 - alpha
        
        return confidence, alpha
    
    def evaluar_modelo_individual(self, model_idx, X_test, y_test=None):
        """Evalúa un modelo individual con métricas de conformal prediction."""
        probs = self.models[model_idx].predict(X_test)
        
        confidences = []
        alphas = []
        
        for prob in probs:
            conf, alpha = self.calcular_confianza_individual(prob)
            confidences.append(conf)
            alphas.append(alpha)
        
        result = {
            'probabilities': probs,
            'confidences': np.array(confidences),
            'alphas': np.array(alphas),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_min': np.min(confidences),
            'confidence_max': np.max(confidences)
        }
        
        # Calcular calibración si tenemos y_test
        if y_test is not None:
            result['calibration'] = self._calcular_calibracion(probs, confidences, y_test)
        
        return result
    
    def evaluar_ensemble(self, X_test, y_test=None):
        """Evalúa el ensemble con métricas de conformal prediction."""
        all_probs = [model.predict(X_test) for model in self.models]
        probs_ensemble = np.mean(all_probs, axis=0)
        
        confidences = []
        alphas = []
        
        for prob in probs_ensemble:
            conf, alpha = self.calcular_confianza_individual(prob)
            confidences.append(conf)
            alphas.append(alpha)
        
        result = {
            'probabilities': probs_ensemble,
            'confidences': np.array(confidences),
            'alphas': np.array(alphas),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_min': np.min(confidences),
            'confidence_max': np.max(confidences)
        }
        
        # Calcular calibración si tenemos y_test
        if y_test is not None:
            result['calibration'] = self._calcular_calibracion(probs_ensemble, confidences, y_test)
        
        # Calcular variabilidad interna entre modelos
        result['variabilidad_interna'] = self._calcular_variabilidad_interna(all_probs)
        
        return result
    
    def _calcular_calibracion(self, probs, confidences, y_true):
        """Calcula Expected Calibration Error (ECE)."""
        # Crear bins por nivel de confianza
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        bin_stats = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Casos en este bin
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
            if i == n_bins - 1:  # Último bin incluye el límite superior
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            
            n_in_bin = np.sum(in_bin)
            
            if n_in_bin > 0:
                # Confianza promedio en el bin
                conf_in_bin = np.mean(confidences[in_bin])
                
                # Accuracy en el bin (predicción correcta si prob>0.5 y y=1, o prob<0.5 y y=0)
                preds_in_bin = (probs[in_bin] > 0.5).astype(int)
                acc_in_bin = np.mean(preds_in_bin == y_true[in_bin])
                
                # Contribución al ECE
                ece += (n_in_bin / len(y_true)) * abs(acc_in_bin - conf_in_bin)
                
                bin_stats.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'n_samples': int(n_in_bin),
                    'confidence_mean': float(conf_in_bin),
                    'accuracy': float(acc_in_bin),
                    'gap': float(abs(acc_in_bin - conf_in_bin))
                })
        
        return {
            'ece': float(ece),
            'bin_stats': bin_stats
        }
    
    def _calcular_variabilidad_interna(self, all_probs):
        """Calcula variabilidad entre los modelos individuales."""
        # Convertir a array [n_models, n_samples]
        probs_array = np.array(all_probs)
        
        # Variabilidad por muestra (std entre modelos)
        std_por_muestra = np.std(probs_array, axis=0)
        
        return {
            'std_mean': float(np.mean(std_por_muestra)),
            'std_std': float(np.std(std_por_muestra)),
            'std_max': float(np.max(std_por_muestra)),
            'std_percentile_95': float(np.percentile(std_por_muestra, 95))
        }