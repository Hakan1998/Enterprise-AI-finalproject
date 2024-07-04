from pipelines import feature_engineering_pipeline, training_pipeline, inference_pipeline

# Funktion zum Löschen von Variablen
def reset_variables(exceptions=None):
    if exceptions is None:
        exceptions = []
    # Liste der Variablen im globalen Namespace abrufen
    all_vars = list(globals().keys())
    # Standardausnahmen hinzufügen
    default_exceptions = ['__name__', '__file__', '__doc__', '__builtins__', 'reset_variables', 'feature_engineering_pipeline', 'training_pipeline', 'inference_pipeline']
    exceptions.extend(default_exceptions)
    # Alle Variablen außer den Ausnahmen löschen
    for var in all_vars:
        if var not in exceptions:
            del globals()[var]


if __name__ == "__main__":
    feature_engineering_pipeline()
    training_pipeline()
    
    reset_variables()


    inference_pipeline()

