SCHEMA = {
    "AGEP": {"dtype": "int64", "min": 0, "max": 95},
    "SEX": {"dtype": "int64", "values": [1, 2]},
    "MSP": {
        "dtype": "str",
        "values": ["1", "2", "3", "4", "5", "6"]
    },
    "HOUSING_TYPE": {"dtype": "int64", "values": [1, 2, 3]},
    "OWN_RENT": {"dtype": "int64", "values": [0, 1, 2]},
    "DENSITY": {"dtype": "float64", "min": 16, "max": 52864},
    "EDU": {
        "dtype": "str",
        "values": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    },
    "PINCP": {
         "dtype": "int65", "min": -9000, "max": 1327000
    },
}
