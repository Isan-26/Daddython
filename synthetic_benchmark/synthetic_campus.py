"""Synthetic campus energy data with known answers, for testing the Track 1 models.

The files have the same columns and the same 12 buildings as the competition data, so any of our notebooks can
run on them unchanged. Every number comes from the simulator below, not from the competition data, and it is a
different campus: a subtropical city with a cool winter and a hot, rainy summer, an academic calendar, and
electric heating as well as cooling.

How one hour of one building is simulated
  weather         daily mean = season + weather swings that last a few days + heatwaves / cold snaps - rain cooling;
                  the temperature follows a daily cycle (warmest at 3 pm); humidity rises at night and with rain
  calendar        weekday / weekend / public holiday (a holiday keeps its normal day name, so a model cannot see it);
                  term, exam weeks, summer break, winter break
  occupancy       capacity x the building type's daily schedule x calendar x reaction to the weather x a random
                  "busy day" factor, plus special events (evening talks, sports matches, open days); Poisson count
  energy (kWh)    always-on equipment + load per person + cooling above 23 C + heating below 15 C (both follow the
                  building's thermal lag and are turned down when the building is closed) + lighting after dark
                  + type extras (hot water in residences, pool heating in sports) + hidden lab-equipment runs
                  + a slow hidden drift (e.g. HVAC efficiency), then meter noise
  previous_usage  the same building's energy one hour earlier (a real time series)

Train rows come from Jan 2024 - Jun 2025 and test rows from Jul - Dec 2025 (the "future"), so the test period has
weather the training data never saw (a stronger heatwave in Aug 2025, a cold snap in Dec 2025) and older sensors
that leave more blanks.

Usage:  python synthetic_campus.py        (writes train.csv, test.csv and test_answers.csv next to this file)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

NUMERIC_COLUMNS = ['temperature', 'humidity', 'occupancy', 'previous_usage']
TRAIN_PERIOD = ('2024-01-01', '2025-06-30 23:00')
TEST_PERIOD = ('2025-07-01', '2025-12-31 23:00')
TRAIN_ROWS, TEST_ROWS = 8000, 3000

# ------------------------------------------------------------------ the buildings
# capacity = people at full use | base = always-on load (kW) | per_person = kW per occupant
# cooling = kW per C above 23 C | heating = kW per C below 15 C | lighting = kW when lit | hvac_24h = never set back
BUILDINGS = pd.DataFrame([
    # id       type              capacity base per_person cooling heating lighting hvac_24h
    ('ADM_A', 'Administration',   120,    16,   0.18,     2.0,    1.3,     7,     False),
    ('BUS_A', 'Business',         260,    20,   0.11,     2.7,    1.6,    11,     False),
    ('BUS_B', 'Business',         320,    25,   0.11,     3.4,    2.0,    13,     False),
    ('ENG_A', 'Engineering',      200,    43,   0.22,     3.2,    1.8,    11,     True),
    ('ENG_B', 'Engineering',      170,    36,   0.20,     2.7,    1.6,     9,     True),
    ('LEC_A', 'LectureHall',      400,    13,   0.06,     4.3,    2.5,    14,     False),
    ('LIB_A', 'Library',          300,    23,   0.11,     3.6,    2.2,    16,     False),
    ('RES_A', 'Residential',      350,    27,   0.08,     2.5,    1.8,    14,     True),
    ('RES_B', 'Residential',      280,    22,   0.08,     2.2,    1.6,    11,     True),
    ('SCI_A', 'Science',          180,    72,   0.23,     4.5,    2.3,    11,     True),
    ('SCI_B', 'Science',          150,    61,   0.22,     3.8,    2.0,     9,     True),
    ('SPT_A', 'Sports',           250,    22,   0.09,     3.6,    1.8,    13,     False),
], columns=['building_id', 'building_type', 'capacity', 'base', 'per_person', 'cooling', 'heating', 'lighting',
            'hvac_24h']).set_index('building_id')

# share of capacity in use by hour: (hour, share) points joined by straight lines, for weekdays and weekends,
# and how busy the building is in each part of the academic calendar
SCHEDULES = {
    'Administration': dict(
        weekday=[(0, .02), (6, .02), (8, .8), (11, .9), (12, .6), (13, .85), (16, .8), (18, .08), (20, .02), (23, .02)],
        weekend=[(0, .02), (9, .02), (10, .08), (13, .08), (14, .02), (23, .02)],
        calendar={'term': 1.0, 'exams': 1.0, 'summer break': 0.8, 'winter break': 0.45}),
    'Business': dict(
        weekday=[(0, .01), (7, .02), (9, .55), (12, .65), (13, .45), (14, .6), (17, .5), (18, .35), (19, .45),
                 (21, .4), (22, .05), (23, .01)],
        weekend=[(0, .01), (8, .01), (9, .25), (13, .25), (14, .05), (23, .01)],
        calendar={'term': 1.0, 'exams': 0.55, 'summer break': 0.25, 'winter break': 0.08}),
    'Engineering': dict(
        weekday=[(0, .05), (7, .06), (9, .6), (12, .65), (13, .5), (14, .65), (18, .45), (20, .2), (23, .08)],
        weekend=[(0, .04), (9, .05), (11, .15), (17, .15), (19, .05), (23, .04)],
        calendar={'term': 1.0, 'exams': 0.7, 'summer break': 0.45, 'winter break': 0.2}),
    'LectureHall': dict(
        weekday=[(0, 0), (7, .02), (8, .55), (11, .8), (12, .35), (13, .75), (16, .6), (17, .3), (19, .08),
                 (21, .02), (23, 0)],
        weekend=[(0, 0), (23, .01)],
        calendar={'term': 1.0, 'exams': 1.05, 'summer break': 0.12, 'winter break': 0.02}),
    'Library': dict(
        weekday=[(0, .03), (7, .03), (8, .15), (11, .4), (15, .55), (18, .45), (21, .4), (23, .15)],
        weekend=[(0, .02), (9, .02), (10, .2), (15, .35), (20, .15), (21, .02), (23, .02)],
        calendar={'term': 1.0, 'exams': 1.7, 'summer break': 0.3, 'winter break': 0.12}),
    'Residential': dict(
        weekday=[(0, .85), (6, .85), (8, .6), (10, .35), (16, .35), (18, .6), (21, .8), (23, .85)],
        weekend=[(0, .85), (9, .8), (12, .6), (17, .6), (21, .8), (23, .85)],
        calendar={'term': 1.0, 'exams': 1.05, 'summer break': 0.25, 'winter break': 0.15}),
    'Science': dict(
        weekday=[(0, .03), (7, .05), (9, .6), (12, .6), (13, .45), (14, .6), (18, .4), (20, .1), (23, .03)],
        weekend=[(0, .02), (10, .08), (16, .08), (18, .02), (23, .02)],
        calendar={'term': 1.0, 'exams': 0.7, 'summer break': 0.55, 'winter break': 0.3}),
    'Sports': dict(
        weekday=[(0, 0), (5, 0), (6, .3), (8, .3), (9, .1), (12, .25), (14, .12), (17, .6), (20, .65), (22, .15),
                 (23, 0)],
        weekend=[(0, 0), (7, 0), (8, .2), (11, .45), (17, .4), (20, .2), (22, 0), (23, 0)],
        calendar={'term': 1.0, 'exams': 0.5, 'summer break': 0.45, 'winter break': 0.25}),
}

HOLIDAYS = pd.to_datetime(['2024-01-01', '2024-03-29', '2024-05-01', '2024-08-17', '2024-12-25', '2025-01-01',
                           '2025-04-18', '2025-05-01', '2025-08-17', '2025-12-25'])
OPEN_DAYS = pd.to_datetime(['2024-04-13', '2024-10-19', '2025-04-12', '2025-10-18'])    # campus open days (Saturdays)
# (first day, last day, change in daily mean temperature at the peak)
WEATHER_EPISODES = [('2024-01-22', '2024-01-26', -5.0, 'cold snap'), ('2024-07-16', '2024-07-20', 4.0, 'heatwave'),
                    ('2025-08-04', '2025-08-11', 6.5, 'heatwave'), ('2025-12-15', '2025-12-19', -5.5, 'cold snap')]


# ------------------------------------------------------------------ weather (one campus weather station)
def simulate_weather(times, rng):
    days = pd.date_range(times[0].normalize(), times[-1].normalize(), freq='D')
    n_days = len(days)
    season_angle = 2 * np.pi * (days.dayofyear.to_numpy() - 200) / 365        # peak of summer in mid-July
    season = 22 + 7 * np.cos(season_angle)                                    # 15 C in January, 29 C in July

    # day-to-day swings that last a few days
    persistence = 0.75
    swings = lfilter([1], [1, -persistence], rng.normal(0, 2.0 * np.sqrt(1 - persistence ** 2), n_days))

    # rainy days: more likely in the summer wet season, and rain spells tend to last several days
    rain_chance = 0.12 + 0.22 * (1 + np.cos(season_angle)) / 2
    rain_day = np.zeros(n_days, bool)
    for d in range(1, n_days):
        rain_day[d] = rng.random() < (min(0.85, rain_chance[d] + 0.35) if rain_day[d - 1] else rain_chance[d])

    episode = np.zeros(n_days)
    episode_name = np.full(n_days, '', dtype=object)
    for first, last, peak, name in WEATHER_EPISODES:
        inside = np.flatnonzero((days >= first) & (days <= last))
        episode[inside] = peak * np.sin(np.pi * (np.arange(len(inside)) + 1) / (len(inside) + 1)) ** 0.5
        episode_name[inside] = name

    daily_mean = season + swings + episode - 2.5 * rain_day
    daily_range = np.where(rain_day, 2.2, 4.5)                                 # cloudy days swing less

    day_number = ((times.normalize() - days[0]).days).to_numpy()
    hour = times.hour.to_numpy()
    smooth_mean = np.interp(day_number + hour / 24, np.arange(n_days) + 0.5, daily_mean)   # no jump at midnight
    temperature = (smooth_mean + daily_range[day_number] * np.cos(2 * np.pi * (hour - 15) / 24)
                   + rng.normal(0, 0.35, len(times)))

    afternoon = np.exp(-0.5 * ((hour - 16) / 3.5) ** 2)                        # showers peak in the afternoon
    rain_strength = rain_day[day_number] * (0.55 + 0.45 * afternoon)
    wet_season = (1 + np.cos(season_angle[day_number])) / 2
    humidity = (60 + 10 * wet_season - 2.6 * (temperature - smooth_mean) + 20 * rain_strength
                + rng.normal(0, 2.5, len(times)))
    return pd.DataFrame({
        'temperature': temperature, 'humidity': np.clip(humidity, 20, 100), 'daily_mean': smooth_mean,
        'rain_day': rain_day[day_number], 'episode': episode_name[day_number],
    }, index=times)


def term_status(dates):
    month_day = dates.month * 100 + dates.day
    status = np.full(len(dates), 'term', dtype=object)
    status[(month_day >= 1221) | (month_day <= 114)] = 'winter break'
    status[(month_day >= 511) & (month_day <= 531)] = 'exams'
    status[(month_day >= 601) & (month_day <= 825)] = 'summer break'
    status[(month_day >= 1206) & (month_day <= 1220)] = 'exams'
    return status


def day_profile(points, hours):
    xs, ys = zip(*points)
    return np.interp(hours, xs, ys)


def hidden_drift(n, rng, persistence=0.998, size=0.05):
    """Slow random change in efficiency (e.g. dirty filters, set-point changes): about +-5%, lasting weeks."""
    return np.exp(lfilter([1], [1, -persistence], rng.normal(0, size * np.sqrt(1 - persistence ** 2), n)))


# ------------------------------------------------------------------ one building, every hour
def simulate_building(building_id, weather, rng):
    spec = BUILDINGS.loc[building_id]
    kind = spec['building_type']
    schedule = SCHEDULES[kind]
    times = weather.index
    n = len(times)
    hour, dates = times.hour.to_numpy(), times.normalize()
    day_number = ((dates - dates[0]).days).to_numpy()
    holiday = dates.isin(HOLIDAYS)
    weekend = (times.dayofweek.to_numpy() >= 5) | holiday
    status = term_status(dates)
    temperature, humidity = weather['temperature'].to_numpy(), weather['humidity'].to_numpy()

    # --- occupancy
    share = np.where(weekend, day_profile(schedule['weekend'], hour), day_profile(schedule['weekday'], hour))
    share = share * pd.Series(status).map(schedule['calendar']).to_numpy()
    event = np.zeros(n, bool)
    if kind == 'LectureHall':
        block = day_number * 12 + hour // 2                                     # each 2-hour class slot fills differently
        share = share * rng.lognormal(0, 0.3, block.max() + 1)[block]
        talk_days = rng.random(day_number.max() + 1) < 0.04                   # evening talks on 4% of term weekdays
        talk = talk_days[day_number] & ~weekend & (status == 'term') & (hour >= 18) & (hour <= 20)
        share, event = share + 0.75 * talk, event | talk
    if kind == 'Library':
        share = share + 0.2 * ((status == 'exams') & (hour < 8))               # open all night in exam weeks
        share = share * np.where(weather['rain_day'], 1.15, 1.0)
    if kind == 'Residential':
        share = share * np.where((temperature > 32) & (hour >= 9) & (hour <= 17), 1.25, 1.0)   # stay in, AC on
    if kind == 'Sports':
        match_days = rng.random(day_number.max() + 1) < 0.35                  # matches on 35% of term Saturdays
        match = (match_days[day_number] & (times.dayofweek.to_numpy() == 5) & (status == 'term')
                 & (hour >= 14) & (hour <= 16))
        share, event = share + 1.1 * match, event | match
        share = share * np.where(weather['rain_day'], 1.2, 1.0)
    if kind not in ('Residential', 'Sports'):
        open_day = dates.isin(OPEN_DAYS) & (hour >= 9) & (hour <= 15)
        share, event = share + 0.35 * open_day, event | open_day
    share = np.minimum(share, 1.6 if kind == 'Sports' else 1.15)            # seats / spectator limit
    building_open = share >= 0.08
    busy_day = rng.lognormal(0, 0.12, day_number.max() + 1)[day_number]
    occupancy = rng.poisson(spec['capacity'] * share * busy_day * rng.lognormal(0, 0.08, n)).astype(float)
    occupancy = np.minimum(occupancy, round(spec['capacity'] * (2.0 if kind == 'Sports' else 1.2)))   # room limit

    # --- energy
    lag = 0.3                                                                  # the building warms/cools slowly
    felt_temperature = lfilter([lag], [1, lag - 1], temperature, zi=[(1 - lag) * temperature[0]])[0]
    hvac = np.where(building_open | spec['hvac_24h'], 1.0, 0.35)               # turned down when closed
    cooling = (spec['cooling'] * np.clip(felt_temperature - 23, 0, None)
               * (1 + 0.015 * np.clip(humidity - 55, 0, None)) * hvac)         # humid air costs extra to cool
    heating = spec['heating'] * np.clip(15 - felt_temperature, 0, None) * hvac
    people = spec['per_person'] * occupancy + 0.007 * occupancy * np.clip(felt_temperature - 20, 0, None)

    sun_angle = np.cos(2 * np.pi * (times.dayofyear.to_numpy() - 5) / 365)     # +1 in early January
    dark = (hour + 0.5 < 6.3 + 0.9 * sun_angle) | (hour + 0.5 > 18.2 - 0.9 * sun_angle)
    if kind == 'Residential':
        lighting = spec['lighting'] * dark * (0.1 + occupancy / spec['capacity'])
    else:
        lighting = spec['lighting'] * dark * np.where(building_open, 1.0, 0.15)

    extras = np.zeros(n)
    if kind == 'Residential':                                                  # showers and cooking
        peak_hours = ((hour >= 6) & (hour <= 8)) | ((hour >= 18) & (hour <= 22))
        extras = 0.055 * occupancy * peak_hours * (1 + 0.05 * np.clip(20 - temperature, 0, None))
    if kind == 'Sports':                                                       # heated pool
        extras = 1.6 * np.clip(26 - weather['daily_mean'].to_numpy(), 0, None)

    equipment_run = np.zeros(n, bool)                                          # hidden: test rigs, furnaces, ...
    equipment = np.zeros(n)
    if kind in ('Engineering', 'Science'):
        for start in np.flatnonzero(rng.random(n) < 0.005):
            hours_on = rng.integers(2, 9)
            equipment[start:start + hours_on] += rng.uniform(14, 40)
            equipment_run[start:start + hours_on] = True

    expected_energy = (spec['base'] + people + cooling + heating + lighting + extras + equipment) * hidden_drift(n, rng)
    energy = np.clip(expected_energy * (1 + rng.normal(0, 0.02, n)) + rng.normal(0, 1.2, n), 1, None).round(1)

    return pd.DataFrame({
        'time': times, 'building_id': building_id, 'building_type': kind, 'hour': hour,
        'day_of_week': times.day_name(), 'month': times.month,
        'temperature': (temperature + rng.normal(0.0, 0.3) + rng.normal(0, 0.2, n)).round(1),   # building's own sensor
        'humidity': np.clip(humidity + rng.normal(0, 1.5) + rng.normal(0, 1.0, n), 20, 100).round(1),
        'occupancy': occupancy,
        'previous_usage': np.r_[np.nan, energy[:-1]],
        'energy_usage': energy, 'expected_energy': expected_energy.round(2),
        'situation': situation_label(equipment_run, event, weather, holiday, status),
    })


def situation_label(equipment_run, event, weather, holiday, status):
    """The first thing that applies, in this order (for checking errors by situation; never given to the model)."""
    checks = [('equipment run', equipment_run), ('special event', event),
              ('heatwave', weather['episode'].to_numpy() == 'heatwave'),
              ('cold snap', weather['episode'].to_numpy() == 'cold snap'),
              ('rain', weather['rain_day'].to_numpy()), ('holiday', holiday), ('exams', status == 'exams'),
              ('break', np.isin(status, ['summer break', 'winter break']))]
    label = np.full(len(status), 'normal', dtype=object)
    for name, applies in reversed(checks):
        label[np.asarray(applies)] = name
    return label


# ------------------------------------------------------------------ sensor blanks
def add_blanks(rows, rng):
    """Blanks come from sensor problems, and sensors get worse after July 2025 (the test period):
    random dropouts, whole days without a building's weather sensor or people counter, meter outages that
    remove previous_usage for a few hours, and humidity sensors that fail in condensation (very humid air)."""
    rows = rows.copy()
    later = (rows['time'] >= TEST_PERIOD[0]).to_numpy()
    n = len(rows)
    blank = {column: np.zeros(n, bool) for column in NUMERIC_COLUMNS}

    for column in NUMERIC_COLUMNS:                                            # random dropouts
        blank[column] |= rng.random(n) < np.where(later, 0.04, 0.015)

    day_key = rows['building_id'] + rows['time'].dt.strftime('%Y-%m-%d')      # whole-day outages
    for columns, rate_before, rate_later in [(['temperature', 'humidity'], 0.01, 0.04), (['occupancy'], 0.01, 0.05)]:
        days = day_key.unique()
        broken = dict(zip(days, rng.random(len(days))))
        draw = day_key.map(broken).to_numpy()
        for column in columns:
            blank[column] |= draw < np.where(later, rate_later, rate_before)

    for _, positions in rows.groupby('building_id').indices.items():          # meter outages lasting a few hours
        positions = np.sort(positions)
        offline = np.zeros(len(positions), bool)
        starts = np.flatnonzero(rng.random(len(positions)) < np.where(later[positions], 0.006, 0.002))
        for start in starts:
            offline[start:start + rng.geometric(1 / 6)] = True
        blank['previous_usage'][positions] |= offline

    blank['humidity'] |= (rows['humidity'].to_numpy() >= 94) & (rng.random(n) < 0.35)

    for column in NUMERIC_COLUMNS:
        rows[column + '_true'] = rows[column]
        rows.loc[blank[column], column] = np.nan
    return rows


# ------------------------------------------------------------------ put it together
def simulate_campus(seed=2026):
    rng = np.random.default_rng(seed)
    times = pd.date_range('2023-12-31', TEST_PERIOD[1], freq='h')            # one warm-up day, dropped below
    weather = simulate_weather(times, rng)
    hourly = pd.concat([simulate_building(b, weather, rng) for b in BUILDINGS.index], ignore_index=True)
    hourly = hourly[hourly['time'] >= TRAIN_PERIOD[0]].sort_values(['building_id', 'time']).reset_index(drop=True)
    return add_blanks(hourly, rng)


def make_files(folder=Path(__file__).resolve().parent, seed=2026):
    rng = np.random.default_rng(seed + 1)
    hourly = simulate_campus(seed)
    train_pool = hourly[hourly['time'].between(*TRAIN_PERIOD)]
    test_pool = hourly[hourly['time'].between(*TEST_PERIOD)]
    train = train_pool.loc[rng.choice(train_pool.index, TRAIN_ROWS, replace=False)]
    test = test_pool.loc[rng.choice(test_pool.index, TEST_ROWS, replace=False)]
    ids = rng.choice(np.arange(1, 100000), TRAIN_ROWS + TEST_ROWS, replace=False)
    train = train.assign(id=[f'SY{i:06d}' for i in ids[:TRAIN_ROWS]])
    test = test.assign(id=[f'SY{i:06d}' for i in ids[TRAIN_ROWS:]])

    inputs = ['id', 'building_id', 'building_type', 'hour', 'day_of_week', 'month'] + NUMERIC_COLUMNS
    as_files = lambda rows: rows.astype({'occupancy': 'Int64'})
    as_files(train)[inputs + ['energy_usage']].to_csv(folder / 'train.csv', index=False)
    as_files(test)[inputs].to_csv(folder / 'test.csv', index=False)
    answers = test[['id', 'energy_usage', 'expected_energy', 'situation', 'time']
                   + [c + '_true' for c in NUMERIC_COLUMNS]].astype({'occupancy_true': 'Int64'})
    answers.to_csv(folder / 'test_answers.csv', index=False)
    return train, test, answers


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    for name in ['train.csv', 'test.csv', 'test_answers.csv']:
        if (folder / name).exists():
            raise SystemExit(f'{name} already exists here; delete it first to regenerate')
    train, test, answers = make_files(folder)
    print(f'wrote train.csv ({len(train):,} rows), test.csv ({len(test):,} rows) and test_answers.csv to {folder}')
