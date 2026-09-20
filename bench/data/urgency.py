"""A designed urgency ladder used as ground truth for the benchmarks.

These are not drawn from a gold-standard corpus. They are written so that the
intended ordering is defensible to a human reader: each rung describes a
situation plainly worse than the one below it. That is enough to measure whether
a ranking method preserves a known order, which is all the benchmark claims.
"""

# Ordered most urgent first. The index is the ground-truth rank.
TICKETS = [
    "The production database is down and no customer can log in.",
    "Payments are failing for every customer at checkout.",
    "Customer data is being returned to the wrong accounts.",
    "The nightly backup has not run for six days.",
    "A security researcher reported an unauthenticated admin endpoint.",
    "Login is broken on Safari, about a fifth of our traffic.",
    "Invoices show the wrong tax amount for European customers.",
    "The mobile app crashes when opening the settings screen.",
    "Search returns stale results for about an hour after an update.",
    "Password reset emails arrive roughly ten minutes late.",
    "The dashboard chart legend overlaps the axis on small screens.",
    "Two customers asked for a dark mode option.",
    "A help article still references last year's pricing.",
    "The footer copyright year is out of date.",
    "Someone suggested reordering the items in the settings menu.",
    "A typo in the marketing page subtitle: 'recieve' should be 'receive'.",
]


def labeled():
    """Return (key, text, true_rank) triples."""
    return [(f"t{i:02d}", text, i + 1) for i, text in enumerate(TICKETS)]
