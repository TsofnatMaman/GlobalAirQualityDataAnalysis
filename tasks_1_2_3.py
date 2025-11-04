import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from exceptions import CountryNotFoundException

pf = pd.read_excel("who_ambient_air_quality_database_version_2024_(v6.1).xlsx", sheet_name="Update 2024 (V6.1)")
pollution=["pm10_concentration", "pm25_concentration", "no2_concentration"]

# ------------------------------- a -------------------------------
def data_viewer():
    print(pf.head())
    print("describe data:\n",pf.describe())
    print("columns of data:\n",pf.columns)
    print("sum of null values:\n",pf.isnull().sum())
    print("types of columns:\n",pf.dtypes)

# ------------------------------- b -------------------------------
# using numpy
def column_stats(col):
    #col = input("Enter column name: ")
    col_values = pf[col].dropna().values
    print(f"mean of {col} values: ",np.mean(col_values))
    print(f"median of {col} values: ",np.median(col_values))
    print(f"max of {col} values: ",np.max(col_values))
    print(f"min of {col} values: ",np.min(col_values))
    print(f"std of {col} values: ",np.std(col_values))

# ------------------------------- c -------------------------------

# 1 - show average of air quality in global - Each infection separately
def f1():
    global_pollution = pf.groupby("year")[pollution].mean()
    plt.figure()
    for idx, p in enumerate(global_pollution):
        plt.bar(global_pollution[p].index + idx*(1 / len(global_pollution)), global_pollution[p].values, label=p, width=1 / len(global_pollution))

    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Pollution")
    plt.title("Pollution vs. Year")
    plt.show()

# 2 - show trend of air quality mean in specify country - average of all infections
def f2(country, city="all"):

    avg_by_year_in_specify_country = (pf[pf["country_name"].str.lower() == country.lower()])
    if avg_by_year_in_specify_country.empty:
        raise CountryNotFoundException(country)

    if city != "all":
        avg_by_year_in_specify_country = avg_by_year_in_specify_country[avg_by_year_in_specify_country["city"].str.lower().str.contains(city.lower())]

    avg_by_year_in_specify_country =  avg_by_year_in_specify_country.groupby("year")[pollution].mean()
    avg_pollution = avg_by_year_in_specify_country.mean(axis=1)

    plt.figure()

    plt.plot(avg_pollution.index, avg_pollution.values, marker="o")

    plt.xlabel("Year")
    plt.ylabel("Air pollutant concentrations")
    plt.title("Air pollution level per year for country: " + country + " in " + city + " city")
    plt.show()

# 3 - show the contex between pm25_concentration and pm10_concentration
def f3():
    plt.figure()
    plt.scatter(pf["pm25_concentration"], pf["pm10_concentration"], marker=".")

    plt.xlabel("PM2.5")
    plt.ylabel("PM10")
    plt.title("PM2.5 vs PM10.")
    plt.show()


# 4 - show the relationship between level of infection vs amount of population
# using seaborn
def f4():
    population_infection = pf.dropna(subset=["pm25_concentration", "population"])

    plt.figure(figsize=(24,16))
    sns.scatterplot(data=population_infection, x="population", y="pm25_concentration", hue="country_name", alpha=0.6)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xscale("log")
    plt.xlabel("Population (log scale)")
    plt.ylabel("PM2.5")
    plt.title("PM2.5 Concentration vs. Population (log scale)")

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        data_viewer()
        column_stats("pm25_concentration")
        f1()
        f2("israel", "tel aviv")
        f2("GermaNy")
        f3()
        f4()
    except Exception as e:
        print(e)