from client import CodeforcesClient


def main():
    client = CodeforcesClient()

    test_contests = {
        1621: "Hello 2022",
        1624: "Codeforces Round 764 (Div. 3)",
        2038: "2024-2025 ICPC NERC Southern and Volga Russian Regional Contest",
    }

    for contest_id, name in test_contests.items():
        print("=" * 70)
        print(f"Contest: {contest_id} — {name}")
        print("=" * 70)

        try:
            changes = client._request(
                "contest.ratingChanges",
                {"contestId": contest_id},
            )

            print("Rows:", len(changes))

            if changes:
                print("First record:")
                print(changes[0])

        except Exception as exc:
            print("ERROR:", exc)


if __name__ == "__main__":
    main()