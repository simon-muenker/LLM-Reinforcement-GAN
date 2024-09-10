import typing

import pytest


@pytest.fixture(scope="session", autouse=True)
def data() -> typing.List[typing.Dict]:
    return [
        {
            "data": "Just finished my first marathon! Feeling exhausted but proud. #RunningGoals",
            "target": "Congratulations! That's an amazing achievement. How long did you train for it?",
        },
        {
            "data": "Does anyone have recommendations for a good sci-fi book series? I'm in need of a new read!",
            "target": "You should try \'The Expanse\' series by James S. A. Corey. It's an epic space opera with great world-building.",
        },
        {
            "data": "Stuck in traffic again. Why did I move to the city? 😫 #CityLife",
            "target": "Have you considered using public transportation or biking? It might save you some stress!",
        },
        {
            "data": "Just adopted the cutest puppy from the shelter! Meet Max! 🐶",
            "target": "Aww, Max is adorable! Congratulations on your new family member. Remember, patience is key during the training phase!",
        },
        {
            "data": "Anyone else excited for the new Marvel movie coming out next week? #MCU",
            "target": "Can't wait! I've already got my tickets for the midnight premiere. Which character are you most looking forward to seeing?",
        },
    ]
