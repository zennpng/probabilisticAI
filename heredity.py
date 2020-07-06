import csv
import itertools
import sys
from numpy import prod

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data/heredity_data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    prob_list = []
    for person in people:
        if people[person]['mother'] == None and people[person]['father'] == None: 
            if person in one_gene:
                probgene = PROBS["gene"][1]
                jointprob = probgene*PROBS["trait"][1][person in have_trait]
                prob_list.append(jointprob)
            elif person in two_genes:
                probgene = PROBS["gene"][2]
                jointprob = probgene*PROBS["trait"][2][person in have_trait]
                prob_list.append(jointprob)
            else:
                probgene = PROBS["gene"][0]
                jointprob = probgene*PROBS["trait"][0][person in have_trait]
                prob_list.append(jointprob)
        else:
            if people[person]['mother'] in one_gene:
                pm = 0.5
                no_pm = 0.5
            elif people[person]['mother'] in two_genes:
                pm = 1 - PROBS["mutation"]
                no_pm = PROBS["mutation"]
            else:
                pm = PROBS["mutation"]
                no_pm = 1 - PROBS["mutation"]
            if people[person]['father'] in one_gene:
                pf = 0.5
                no_pf = 0.5
            elif people[person]['father'] in two_genes:
                pf = 1 - PROBS["mutation"]
                no_pf = PROBS["mutation"]
            else:
                pf = PROBS["mutation"]
                no_pf = 1 - PROBS["mutation"]
            if person in one_gene:
                probgene = no_pm*pf+pm*no_pf
                jointprob = probgene*PROBS["trait"][1][person in have_trait]
            elif person in two_genes:
                probgene = pm*pf
                jointprob = probgene*PROBS["trait"][2][person in have_trait]
            else:
                probgene = no_pm*no_pf
                jointprob = probgene*PROBS["trait"][0][person in have_trait]
            prob_list.append(jointprob)

    p = prod(prob_list)
    return p
    

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for people in probabilities:
        if people in one_gene:
            probabilities[people]['gene'][1] += p
        elif people in two_genes:
            probabilities[people]['gene'][2] += p
        else:
            probabilities[people]['gene'][0] += p
        if people in have_trait:
            probabilities[people]['trait'][True] += p
        else:
            probabilities[people]['trait'][False] += p
    return None


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for people in probabilities:
        for gt in probabilities[people]:
            dic = probabilities[people][gt]
            norm_constant = 1/sum(dic.values())
            for i in dic:
                dic[i] = dic[i]*norm_constant


if __name__ == "__main__":
    main()
