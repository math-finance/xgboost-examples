__CONST = {
    'Mwh to Therm': 34.12,
    'Therm/Btu': 0.00001,
    'MJ per Kcal': 0.004187,
    'MJ/Btu': 0.001055056,

    'Calorific Value of Coal (Kcal/KG)': 6000,
    'Calorific Value of HFO (MJ/KG)': 40000,
    'Calorific Value of Lignite (MJ/KG)': 9,

    'Carbon Content Lignite (KG CO2/MJ)': 0.11,
    'Carbon Content Gas (KG/GJ)': 0.051,
    'Carbon Content Coal (KG CO2/MJ)': 0.095,
    'Carbon Content Oil (Kg/GJ)': 0.073,
}
heatrate_de_100 = 3412.15


def calc_coal_margin(api2_eur, eua, eff, voc):
    heatrate = heatrate_de_100 / eff
    fuel_cost_eur = api2_eur
    fuel_cost_mwh = fuel_cost_eur / (__CONST['Calorific Value of Coal (Kcal/KG)'] * __CONST['MJ per Kcal']) * __CONST['MJ/Btu'] * heatrate
    carbon_cost_mwh = eua * __CONST['Carbon Content Coal (KG CO2/MJ)'] * __CONST['MJ/Btu'] * heatrate
    return fuel_cost_mwh + carbon_cost_mwh + voc

def calc_gas_margin(ttf, eua, eff, voc):
    heatrate = heatrate_de_100 / eff
    fuel_cost = ttf * (1/__CONST['Mwh to Therm']) * 1000 * __CONST['Therm/Btu'] * heatrate
    carbon_cost = eua * __CONST['Carbon Content Gas (KG/GJ)'] * __CONST['MJ/Btu'] *heatrate
    return fuel_cost + carbon_cost + voc

def calc_lignite_margin(eua, eff, voc):
    heatrate = heatrate_de_100 / eff
    fuel_oil_eur = 5.25
    fuel_cost = fuel_oil_eur / __CONST['Calorific Value of Lignite (MJ/KG)'] * __CONST['MJ/Btu'] * heatrate
    carbon_cost = eua * __CONST['Carbon Content Lignite (KG CO2/MJ)'] * __CONST['MJ/Btu'] *heatrate
    return fuel_cost + carbon_cost + voc

def calc_oil_margin(eua, eurusd, eff, voc):
    heatrate = heatrate_de_100 / eff
    fuel_cost_eur = 650 / eurusd
    fuel_cost_mwh = fuel_cost_eur * 1000 / __CONST['Calorific Value of HFO (MJ/KG)'] * __CONST['MJ/Btu'] * heatrate
    carbon_cost_mwh = eua * __CONST['Carbon Content Oil (Kg/GJ)'] * __CONST['MJ/Btu'] * heatrate
    return fuel_cost_mwh + carbon_cost_mwh + voc
