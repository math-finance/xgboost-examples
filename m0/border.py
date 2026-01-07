__FB_BORDERS = {('DE', 'AT'): '2018-10-01',
                ('DE', 'NL'): '2018-10-01',
                ('DE', 'FR'): '2018-10-01',
                ('FR', 'BE'): '2018-10-01',
                ('BE', 'NL'): '2018-10-01',
                ('BE', 'ALBE'): '2020-11-21',
                ('DE', 'ALDE'): '2020-11-21',
                ('ALBE', 'ALDE'): '2020-11-21',
                ('DE', 'PL'): '2022-06-09',
                ('DE', 'CZ'): '2022-06-09',
                ('CZ', 'PL'): '2022-06-09',
                ('CZ', 'SK'): '2022-06-09',
                ('AT', 'CZ'): '2022-06-09',
                ('PL', 'SK'): '2022-06-09',
                ('SK', 'HU'): '2022-06-09',
                ('AT', 'HU'): '2022-06-09',
                ('HU', 'SI'): '2022-06-09',
                ('HU', 'RO'): '2022-06-09',
                ('HU', 'HR'): '2022-06-09',
                ('SI', 'AT'): '2022-06-09',
                ('SI', 'HR'): '2022-06-09'}


__ALL_BORDERS = [('DE', 'PL'), ('DE', 'CZ'), ('DE', 'DK1'), ('DE', 'DK2'), ('DE', 'SE4'), ('DE', 'NO2'), ('DE', 'CH'),
           ('DE', 'FR'), ('DE', 'NL'), ('DE', 'AT'), ('BE', 'ALBE'), ('DE', 'ALDE'), ('ALBE', 'ALDE'), ('FR', 'BE'), ('BE', 'NL'),
           ('CZ', 'PL'), ('CZ', 'SK'), ('AT', 'CZ'), ('PL', 'SK'), ('SK', 'HU'), ('AT', 'HU'), ('HU', 'RO'),('SI', 'AT'),
           ('SI', 'HU'), ('HR', 'HU'), ('HR', 'SI'), ('FR', 'UK'), ('FR', 'ES'), ('FR', 'CH'), ('FR', 'IT'), ('NL', 'NO2')]

def get_fb_borders_dates():
    # it might be much simpler if we just assume power trade only start when PTDF are fully
    return dict([(link, max(__FB_BORDERS.values()))for link in __FB_BORDERS])

def get_all_borders():
    return tuple(__ALL_BORDERS)
