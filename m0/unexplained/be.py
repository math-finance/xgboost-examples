from edge.models.m0.unexplained import train_unexplained_region_model, calculate_unexplained_region

def calculate_unexplained_be(scenario_path, training=False):
    trained_model_path = "/tmp/unexplained_be_shape_library_and_model.pkl"
    feature_cols = ['cons', "tt_avg", "tt_min", "tt_max", 'wnd', 'spv', "day_of_week", 'month']
    
    if training is True:
        train_unexplained_region_model(region='be', scenario_path=scenario_path, model_path=trained_model_path, feature_cols=feature_cols, has_pumped_store=True)
    
    rslt = calculate_unexplained_region(region='be', scenario_path=scenario_path, trained_model_path=trained_model_path, feature_cols=feature_cols)

    return rslt.round(0).astype(int)
