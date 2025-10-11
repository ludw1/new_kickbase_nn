"""Transform the input JSON data into a format suitable for training."""

import json


def transform_data(input_file: str) -> tuple[list[list[float]], list[dict], int]:
    with open(input_file, "r") as f:
        data = json.load(f)
    # Expects {PlayerName: {player_info: {...}, market_value: {"it": [values]}, ...}}
    # Need to transform it into [[player1_values], [player2_values], ...] and static covariates
    transformed_data = []
    static_covariates = []
    first_date = 70000  # Example Julian date for reference in the far future
    for player, metrics in data.items():
        player_values = []
        market_values = metrics.get("market_value", {}).get("it", [])
        if market_values:
            first_date = min(first_date, market_values[0].get("dt", first_date))
            player_values.extend(
                point.get("mv") for point in market_values if point.get("mv") != 0
            )
        if len(player_values) < 365:
            continue
        elif all(v == 500000 for v in player_values):
            continue
        low_price_count = sum(1 for v in player_values if v == 0 or v == 500000)
        if low_price_count / len(player_values) > 0.3:
            continue

        # Extract static covariates
        player_info = metrics.get("player_info", {})
        static_cov = {
            "team_id": player_info.get("team_id", 0),
            "pos": player_info.get("pos", 0),
        }

        # We filter players with less than 365 days of data, only 0 or 500000 values or more than 30% of such values
        transformed_data.append(player_values)
        static_covariates.append(static_cov)

    return transformed_data, static_covariates, first_date
