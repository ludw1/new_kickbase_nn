"""Transform the input JSON data into a format suitable for training."""

import json
from typing import Tuple


def transform_data(
    input_file: str,
) -> dict[str, Tuple[list[float], dict[str, int], int]]:
    """Takes in the player data file and returns time series data and static covariate data keyed by player.

    Args:
        input_file (str): Input file path.

    Returns:
        dict[str, Tuple[list[float], dict[str, int], int]]: Dictionary with player market values and covariates, first day value.
    """
    with open(input_file, "r") as f:
        data = json.load(f)
    # Expects {PlayerName: {player_info: {...}, market_value: {"it": [values]}, ...}}
    # Need to transform it into [[player1_values], [player2_values], ...] and static covariates
    transformed_data = {}
    first_date = 19983  # Example Julian date for reference
    for player, metrics in data.items():
        player_values = []
        market_values = metrics.get("market_value", {}).get("it", [])
        if market_values:
            first_date = market_values[0].get("dt", first_date)
            player_values.extend(
                point.get("mv") for point in market_values
            )
        if len(player_values) == 0:
            continue
        # Extract static covariates
        player_info = metrics.get("player_info", {})
        static_cov = {
            "team_id": player_info.get("team_id", 0),
            "pos": player_info.get("pos", 0),
        }

        # We filter players with less than 365 days of data, only 0 or 500000 values or more than 30% of such values
        transformed_data[player] = (player_values, static_cov, first_date)

    return transformed_data
