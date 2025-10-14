import logging
import aiohttp
import asyncio
import json
import os
from get_data.models import TeamResponse, PlayerMarketValueResponse, PlayerInfo, PlayerPerformanceEntry
from get_data.auth import login
from config import PipelineConfig

logger = logging.getLogger(__name__)


class GetPlayerData:
    def __init__(self, token: str, session: aiohttp.ClientSession):
        self.token = token
        self._session = session  # So we can use the same session for multiple requests

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._session.close()

    @property
    def session(self):
        if self._session is None:
            raise RuntimeError("Session is not initialized.")
        return self._session

    async def get_player_ids(self, competition: int) -> list[TeamResponse]:
        """Get all teams and players within those teams for a competition.

        Args:
            competition (int): The competition ID, 1 is Bundesliga, 2 is 2. Bundesliga.

        Returns:
            list[TeamResponse]: A list of team responses containing player data.
        """
        logger.info("Fetching teams data from API.")
        url = "https://api.kickbase.com/v4/competitions/{competition}/teams/{team_id}/teamprofile"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cookie": f"kkstrauth={self.token};",
        }
        all_teams = []

        async def fetch_team(team_id: int) -> TeamResponse:
            team_data = TeamResponse(tid=0, tn="", it=[])
            logger.info(f"Fetching data for team {team_id}")
            try:
                async with self.session.get(
                    url.format(team_id=team_id, competition=competition),
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    logger.info(f"Data for team {team_id} fetched successfully.")
                    team_data = TeamResponse(**await response.json())

            except Exception as e:
                logger.info(f"Error fetching data for team {team_id}: {e}")
            return team_data

        tasks = [
            fetch_team(team_id) for team_id in range(2, 150)
        ]  # We dont know the exact range
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for team_data in results:
            logger.info(f"Processing team data: {team_data}")
            if isinstance(team_data, BaseException) or not team_data.it:
                continue
            else:
                all_teams.append(team_data)
        logger.info(f"Found {len(all_teams)} teams with players.")
        return all_teams

    async def get_player_data(
        self, player_id: str
    ) -> tuple[PlayerMarketValueResponse, str]:
        """Get player data market value history.

        Args:
            player_id (str): The ID of the player.

        Returns:
            tuple[PlayerMarketValueResponse, str]: A tuple containing the player market value response and player ID.
        """
        url = f"https://api.kickbase.com/v4/competitions/1/players/{player_id}/marketvalue/365"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cookie": f"kkstrauth={self.token};",
        }
        player_data = PlayerMarketValueResponse(it=[])
        logger.info(f"Fetching data for player {player_id}")
        try:
            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                logger.info(f"Data for player {player_id} fetched successfully.")
                player_data = PlayerMarketValueResponse(**await response.json())
        except Exception as e:
            logger.info(f"Error fetching data for player {player_id}: {e}")
        return player_data, player_id

    async def get_player_info(self, player_id: str) -> tuple[PlayerInfo, str]:
        """Get player information.

        Args:
            player_id (str): The ID of the player.

        Returns:
            tuple[PlayerInfo, str]: A tuple containing the player info response and player ID.
        """
        url = f"https://api.kickbase.com/v4/competitions/1/players/{player_id}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cookie": f"kkstrauth={self.token};",
        }
        player_info = PlayerInfo(i=0, fn="", ln="", shn=0, tid=0, tn="", st=0, stl=[], pos=0, iposl=False, tp=0, ap=0, g=0, a=0)
        logger.info(f"Fetching info for player {player_id}")
        try:
            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                logger.info(f"Info for player {player_id} fetched successfully.")
                player_info = PlayerInfo(**await response.json())
        except Exception as e:
            logger.info(f"Error fetching info for player {player_id}: {e}")
        return player_info, player_id

    async def get_player_performance(self, player_id: str) -> tuple[PlayerPerformanceEntry, str]:
        """Get player performance history.

        Args:
            player_id (str): The ID of the player.

        Returns:
            tuple[PlayerPerformanceEntry, str]: A tuple containing the player performance response and player ID.
        """
        url = f"https://api.kickbase.com/v4/competitions/1/players/{player_id}/performance"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cookie": f"kkstrauth={self.token};",
        }
        player_performance = PlayerPerformanceEntry(it=[])
        logger.info(f"Fetching performance for player {player_id}")
        try:
            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                logger.info(f"Performance for player {player_id} fetched successfully.")
                player_performance = PlayerPerformanceEntry(**await response.json())
        except Exception as e:
            logger.info(f"Error fetching performance for player {player_id}: {e}")
        return player_performance, player_id

    async def get_all_player_data(
        self, teams: list[TeamResponse]
    ) -> dict[str, dict]:
        """Get player data for all players in a team.

        Args:
            teams (list[TeamResponse]): A list of team responses containing player data.

        Returns:
            dict[str, dict]: A dictionary of player data containing market value, info, and performance, keyed to player id.
        """
        all_player_data = {}
        all_player_ids = [player.i for team in teams for player in team.it]

        # Create tasks for all three types of data
        market_tasks = [self.get_player_data(player_id) for player_id in all_player_ids]
        info_tasks = [self.get_player_info(player_id) for player_id in all_player_ids]
        performance_tasks = [self.get_player_performance(player_id) for player_id in all_player_ids]

        # Execute all tasks
        market_responses = await asyncio.gather(*market_tasks, return_exceptions=True)
        info_responses = await asyncio.gather(*info_tasks, return_exceptions=True)
        performance_responses = await asyncio.gather(*performance_tasks, return_exceptions=True)

        # Process market value responses
        market_data = {}
        for response in market_responses:
            if isinstance(response, BaseException):
                continue
            player_market_value, player_id = response
            if player_market_value.it:
                market_data[player_id] = player_market_value

        # Process info responses
        info_data = {}
        for response in info_responses:
            if isinstance(response, BaseException):
                continue
            player_info, player_id = response
            info_data[player_id] = player_info

        # Process performance responses
        performance_data = {}
        for response in performance_responses:
            if isinstance(response, BaseException):
                continue
            player_performance, player_id = response
            if player_performance.it:
                performance_data[player_id] = player_performance

        # Combine all data
        for player_id in all_player_ids:
            all_player_data[player_id] = {
                'market_value': market_data.get(player_id),
                'info': info_data.get(player_id),
                'performance': performance_data.get(player_id)
            }

        return all_player_data


async def get_api_data():
    user = login()
    token = user.tkn
    all_player_data = {}
    file_player_data = {}
    async with aiohttp.ClientSession() as session:
        player_data = GetPlayerData(token, session)
        erste = await player_data.get_player_ids(1)  # Bundesliga
        zweite = await player_data.get_player_ids(2)  # 2. Bundesliga
        result = erste + zweite
        player_data_lookup = {
            player.i: {
                "team_id": team.tid,
                "team_name": team.tn,
                "name": player.n,
                "pos": player.pos,
            }
            for team in result
            for player in team.it
        }
        # Ensure data directory exists
        os.makedirs(os.path.dirname(PipelineConfig.TEAMS_FILE) or ".", exist_ok=True)

        with open(PipelineConfig.TEAMS_FILE, "w") as f:
            writable_result = [
                team.model_dump() for team in result
            ]  # Need to convert to dicts
            json.dump(writable_result, f, indent=4)
        logger.info(f"Total teams fetched: {len(result)}")
        logger.info(f"Teams data saved to: {PipelineConfig.TEAMS_FILE}")

        all_player_data = await player_data.get_all_player_data(result)
        logger.info(f"Total players fetched: {len(all_player_data)}")
        for player_id, player_data_dict in all_player_data.items():
            file_player_data[
                player_data_lookup.get(player_id, {}).get("name", "") + "_" + player_id
            ] = {  # Use name and id to handle duplicate names
                "player_info": player_data_lookup.get(player_id, {}),
                "market_value": player_data_dict['market_value'].model_dump() if player_data_dict['market_value'] else None,
                "info": player_data_dict['info'].model_dump() if player_data_dict['info'] else None,
                "performance": player_data_dict['performance'].model_dump() if player_data_dict['performance'] else None,
            }
        with open(PipelineConfig.DATA_FILE, "w") as f:
            json.dump(file_player_data, f, indent=4)
        logger.info(f"Player data saved to: {PipelineConfig.DATA_FILE}")
