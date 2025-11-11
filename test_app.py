import pytest
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

# URL of the deployed application
URL = "http://test-env.eba-iwxxdzdm.ca-central-1.elasticbeanstalk.com/predict"

# Test cases
REAL_NEWS_1 = {
    "message": "Toronto's newest park is now open but don't get too attached to it",
}

REAL_NEWS_2 = {
    "message": "One of Matty Matheson’s Toronto restaurants closes less than a year after opening",
}

FAKE_NEWS_1 = {
    "message": "Aliens have visited the world today",
}

FAKE_NEWS_2 = {
    "message": "BREAKING: new disease wipes out half the world",
}

TEST_CASES = {
    "real_news_1": REAL_NEWS_1,
    "real_news_2": REAL_NEWS_2,
    "fake_news_1": FAKE_NEWS_1,
    "fake_news_2": FAKE_NEWS_2,
}

# Functional Tests
def test_real_news_1():
    response = requests.post(URL, json=REAL_NEWS_1)
    print(response.json())
    assert response.json()["label"] == "REAL"

def test_real_news_2():
    response = requests.post(URL, json=REAL_NEWS_2)
    print(response.json())
    assert response.json()["label"] == "REAL"

def test_fake_news_1():
    response = requests.post(URL, json=FAKE_NEWS_1)
    print(response.json())
    assert response.json()["label"] == "FAKE"

def test_fake_news_2():
    response = requests.post(URL, json=FAKE_NEWS_2)
    print(response.json())
    assert response.json()["label"] == "FAKE"

# Performance Tests
def run_performance_test():
    if not os.path.exists('performance_results'):
        os.makedirs('performance_results')

    all_latencies = {}

    for name, data in TEST_CASES.items():
        latencies = []
        for _ in range(100):
            start_time = time.time()
            requests.post(URL, json=data)
            end_time = time.time()
            latencies.append(end_time - start_time)
        
        all_latencies[name] = latencies
        df = pd.DataFrame({"latency": latencies})
        df.to_csv(f"performance_results/{name}_latencies.csv", index=False)

    # Generate boxplot
    df_all = pd.DataFrame(all_latencies)
    plt.figure(figsize=(10, 6))
    df_all.boxplot()
    plt.title("API Latency for Different News Types")
    plt.ylabel("Latency (seconds)")
    plt.savefig("performance_results/latency_boxplot.png")

    # Calculate and print average latencies
    for name, latencies in all_latencies.items():
        print(f"Average latency for {name}: {sum(latencies) / len(latencies):.4f} seconds")

if __name__ == "__main__":
    # Run functional tests with pytest
    pytest.main(["-v", __file__])
    
    # Run performance tests
    run_performance_test()
