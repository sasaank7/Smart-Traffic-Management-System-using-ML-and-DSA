"""
City Traffic Infrastructure Integration

Connects to city traffic management systems and sensors:
- Traffic signal controllers
- Loop detectors
- CCTV cameras
- Variable message signs
- Emergency vehicle preemption systems
"""
import asyncio
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrafficSensorReading:
    """Traffic sensor data structure"""
    sensor_id: str
    timestamp: datetime
    vehicle_count: int
    average_speed: float
    occupancy: float
    lane: int
    sensor_type: str  # loop, radar, camera


class CityTrafficIntegration:
    """
    Integration with city traffic management systems

    Supports multiple protocols:
    - NTCIP (National Transportation Communications for ITS Protocol)
    - SNMP (Simple Network Management Protocol)
    - REST APIs
    - MQTT for IoT sensors
    """

    def __init__(self, config: dict):
        self.config = config
        self.session = None
        self.sensors = {}
        self.signals = {}

    async def connect(self):
        """Establish connection to city systems"""
        self.session = aiohttp.ClientSession()
        logger.info("Connected to city traffic infrastructure")

    async def disconnect(self):
        """Close connections"""
        if self.session:
            await self.session.close()

    async def fetch_sensor_data(self, sensor_id: str) -> Optional[TrafficSensorReading]:
        """
        Fetch data from traffic sensor

        Args:
            sensor_id: Sensor identifier

        Returns:
            TrafficSensorReading or None
        """
        # Example implementation for REST API
        api_url = f"{self.config['api_base_url']}/sensors/{sensor_id}"

        try:
            async with self.session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()

                    reading = TrafficSensorReading(
                        sensor_id=sensor_id,
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        vehicle_count=data['vehicle_count'],
                        average_speed=data['average_speed'],
                        occupancy=data['occupancy'],
                        lane=data.get('lane', 1),
                        sensor_type=data.get('type', 'loop'),
                    )

                    return reading

        except Exception as e:
            logger.error(f"Error fetching sensor {sensor_id}: {e}")

        return None

    async def fetch_all_sensors(self) -> List[TrafficSensorReading]:
        """Fetch data from all registered sensors"""
        tasks = [
            self.fetch_sensor_data(sensor_id)
            for sensor_id in self.sensors.keys()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        readings = [r for r in results if isinstance(r, TrafficSensorReading)]

        logger.info(f"Fetched {len(readings)} sensor readings")
        return readings

    async def control_signal(
        self,
        signal_id: str,
        direction: str,
        duration: int,
        priority: bool = False
    ):
        """
        Send control command to traffic signal

        Args:
            signal_id: Traffic signal ID
            direction: Direction (north_south, east_west)
            duration: Green light duration in seconds
            priority: Emergency priority flag
        """
        api_url = f"{self.config['api_base_url']}/signals/{signal_id}/control"

        payload = {
            'direction': direction,
            'duration': duration,
            'priority': priority,
            'timestamp': datetime.utcnow().isoformat(),
        }

        try:
            async with self.session.post(api_url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"Signal {signal_id} controlled: {direction} for {duration}s")
                    return True
                else:
                    logger.error(f"Failed to control signal {signal_id}: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Error controlling signal {signal_id}: {e}")
            return False

    async def emergency_preemption(
        self,
        route_signal_ids: List[str],
        vehicle_id: str
    ):
        """
        Activate emergency vehicle preemption

        Gives green lights along emergency vehicle route

        Args:
            route_signal_ids: List of signal IDs along route
            vehicle_id: Emergency vehicle identifier
        """
        logger.info(f"Activating emergency preemption for {vehicle_id}")

        tasks = []
        for signal_id in route_signal_ids:
            task = self.control_signal(
                signal_id=signal_id,
                direction="all",  # Typically handled by signal controller
                duration=60,
                priority=True
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r)
        logger.info(f"Emergency preemption: {successful}/{len(route_signal_ids)} signals activated")

        return successful == len(route_signal_ids)

    async def update_vms(self, message: str, locations: List[str]):
        """
        Update Variable Message Signs (VMS)

        Args:
            message: Message to display
            locations: VMS locations
        """
        api_url = f"{self.config['api_base_url']}/vms/update"

        payload = {
            'message': message,
            'locations': locations,
            'priority': 'high',
        }

        try:
            async with self.session.post(api_url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"VMS updated: {message} at {len(locations)} locations")
                    return True

        except Exception as e:
            logger.error(f"Error updating VMS: {e}")

        return False

    async def process_cctv_feed(self, camera_id: str):
        """
        Process CCTV camera feed

        Args:
            camera_id: Camera identifier

        Returns:
            Vehicle count and classifications
        """
        # This would interface with video stream
        # For now, return sample data
        return {
            'camera_id': camera_id,
            'timestamp': datetime.utcnow().isoformat(),
            'vehicle_count': 15,
            'classifications': {
                'car': 12,
                'truck': 2,
                'bus': 1,
            }
        }

    def register_sensor(self, sensor_id: str, config: dict):
        """Register a traffic sensor"""
        self.sensors[sensor_id] = config
        logger.info(f"Registered sensor: {sensor_id}")

    def register_signal(self, signal_id: str, config: dict):
        """Register a traffic signal"""
        self.signals[signal_id] = config
        logger.info(f"Registered signal: {signal_id}")


# Example usage
async def main():
    """Example integration"""
    config = {
        'api_base_url': 'http://city-traffic-api.local',
        'auth_token': 'your-api-token',
    }

    integration = CityTrafficIntegration(config)
    await integration.connect()

    # Register sensors
    for i in range(1, 11):
        integration.register_sensor(f"sensor_{i}", {
            'type': 'loop',
            'location': f'intersection_{i}',
        })

    # Fetch sensor data
    readings = await integration.fetch_all_sensors()
    print(f"Fetched {len(readings)} sensor readings")

    # Control signal
    await integration.control_signal(
        signal_id="signal_1",
        direction="north_south",
        duration=45
    )

    # Emergency preemption example
    await integration.emergency_preemption(
        route_signal_ids=['signal_1', 'signal_2', 'signal_3'],
        vehicle_id='ambulance_101'
    )

    await integration.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
