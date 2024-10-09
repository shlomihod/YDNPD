SCHEMA = {
    "AGEP": {"dtype": "int64", "min": 0, "max": 95},
    "SEX": {"dtype": "int64", "values": [1, 2]},
    "MSP": {
        "has_null": True, "null_value": "N", "dtype": "object",
        "values": ["N", "1", "2", "3", "4", "5", "6"]
    },
    "HOUSING_TYPE": {"dtype": "int64", "values": [1, 2, 3]},
    "OWN_RENT": {"dtype": "int64", "values": [0, 1, 2]},
    "DENSITY": {"dtype": "float64", "min": 16, "max": 52864},
    "EDU": {
        "has_null": True, "null_value": "N", "dtype": "object",
        "values": ["N", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    },
    "PINCP": {
        "has_null": True, "null_value": "N", "dtype": "object",
        "min": -9000, "max": 1327000
    },
}

CENSUS_PROMPT = [
    {
      "role": "system",
      "content": "You are an expert on demographic and census data."
    },
    {
      "role": "user",
      "content": "You are a helpful assistant to experts experts on demographic and census data. Our goal is to construct a causal graph between the following variables.\n"
    }
]
