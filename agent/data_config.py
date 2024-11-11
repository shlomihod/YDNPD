# https://pages.nist.gov/privacy_collaborative_research_cycle/pages/participate.html
# https://github.com/usnistgov/SDNist/blob/main/nist%20diverse%20communities%20data%20excerpts/data_dictionary.json

CENSUS_DATASET_METADATA = {
  "domain": "demographic and census data",
  "schema": {
    "SEX": {
      "dtype": "int64",
      "values": [
        1,
        2
      ]
    },
    "MSP": {
      "description": "Marital Status",
      "dtype": "int64",
      "values": {
        1: "Now married, spouse present",
        2: "Now Married, spouse absent",
        3: "Widowed",
        4: "Divorced",
        5: "Separated",
        6: "Never married"
      }
    },
    "RAC1P": {
      "description": "Person's Race",
      "dtype": "int64",
      "values": {
        1: "White alone",
        2: "Black or African American alone",
        3: "American Indian alone",
        4: "Alaska Native alone",
        5: "American Indian and Alaska Native tribes specified; or American Indian or Alaska Native, not specified and no other races",
        6: "Asian alone",
        7: "Native Hawaiian and Other Pacific Islander alone",
        8: "Some Other Race alone",
        9: "Two or More Races"
      }
    },
  "HOUSING_TYPE": {
      "description": "Housing unit or group quarters",
      "dtype": "int64",
      "values": {
        1: "Housing Unit",
        2: "Institutional Group Quarters",
        3: "Non-institutional Group Quarters"
      }
    },
    "OWN_RENT": {
      "description": "Housing unit rented or owned",
      "dtype": "int64",
      "values": {
        0: "Group quarters",
        1: "Own housing unit",
        2: "Rent housing unit"
      }
    },
    "PINCP_DECILE": {
      "description": "Person's total income rank (with respect to their state) discretized into 10% bins.",
      "dtype": "int64",
      "values": {
        9: "90th percentile",
        8: "80th percentile",
        7: "70th percentile",
        6: "60th percentile",
        5: "50th percentile",
        4: "40th percentile",
        3: "30th percentile",
        2: "20th percentile",
        1: "10th percentile",
        0: "0th percentile"
      }
    },
    "EDU": {
      "description": "Educational attainment",
      "dtype": "int64",
      "values": {
        1: "No schooling completed",
        2: "Nursery school, Preschool, or Kindergarten",
        3: "Grade 1 to grade 8",
        4: "Grade 9 to grade 12, no diploma",
        5: "High School diploma",
        6: "GED",
        7: "Some College, no degree",
        8: "Associate degree",
        9: "Bachelors degree",
        10: "Masters degree",
        11: "Professional degree",
        12: "Doctorate degree"
      }
    }
  }
}

OLD_CENSUS_DATASET_METADATA = {
  "domain": "demographic and census data",
  "schema": {
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
}
