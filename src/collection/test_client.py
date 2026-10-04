from client import CodeforcesClient


def main():
    client = CodeforcesClient()

    print("Fetching user info...")
    user = client.get_user_info("tourist")
    print(user[0])

    print("\nFetching rating history...")
    ratings = client.get_rating_history("tourist")
    print(f"Rating records: {len(ratings)}")
    print(ratings[:2])

    print("\nFetching submissions...")
    submissions = client.get_submissions(
        "tourist",
        from_=1,
        count=1000,
    )
    print(f"Submissions returned: {len(submissions)}")
    print("Newest submission ID:", submissions[0]["id"])
    print("Oldest submission ID:", submissions[-1]["id"])
    for submission in submissions:
        print(submission)


if __name__ == "__main__":
    main()