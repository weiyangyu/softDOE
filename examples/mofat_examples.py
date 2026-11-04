"""Run MOFAT-style examples: python examples/mofat_examples.py."""
import numpy as np
import softDOE


def friedman(x):
    return (10*np.sin(np.pi*x[0]*x[1]) + 20*(x[2]-0.5)**2
            + 10*x[3] + 5*x[4])


def main():
    # MOFAT: result <- soft(p=2, l=4); result$design
    result = softDOE.soft(p=2,l=4,seed=123,backend="cpu")
    print("SOFT (12 x 2):\n",result["design"])
    print("Q:",result["Q"],"normalized Q:",result["normalized_Q"])
    strict = softDOE.soft(p=2,l=4,structure="strict",seed=123,backend="cpu")
    print("Strict SOFT:",strict["design"].shape)

    # MOFAT: result <- a_soft(p=2, l=4, n_aug=8)
    augmented = softDOE.a_soft(p=2,l=4,n_aug=8,seed=123,backend="cpu")
    print("A-SOFT:",augmented["design"].shape)
    print("Final augmented Q:",augmented["Q"],"normalized Q:",augmented["normalized_Q"])
    existing = result["design"]
    extended = softDOE.a_soft(n_aug=8,ofat_design=existing,backend="cpu")
    np.testing.assert_array_equal(extended["design"][:len(existing)],existing)
    print("Supplied-design augmentation:",extended["design"].shape)

    # MOFAT: qmc <- qmc_generate(n=1000, p=2)
    qmc = softDOE.qmc_generate(n=1000,p=2)
    print("Sobol + corners:",qmc.shape)
    assert qmc.shape == (1004,2)

    # MOFAT: Q(1, 1/(2*4)^2, design, qmc)
    scores = softDOE.Q(1,1/(2*4)**2,result["design"],qmc,backend="cpu")
    print("Q on the supplied evaluation set:",scores)

    # MOFAT's Friedman screening example, using the SOFT design generator.
    design = softDOE.soft(p=10,l=4,seed=123,backend="cpu")["design"]
    y = np.array([friedman(row) for row in design])
    screening = softDOE.measure(design,y)
    print("Friedman mustar:",screening["mustar"])
    print("Friedman total Sobol indices:",screening["t"])
    np.testing.assert_array_equal(screening["mustar"][5:],np.zeros(5))
    np.testing.assert_array_equal(screening["t"][5:],np.zeros(5))
    print("Backends:",softDOE.backend_info())


if __name__ == "__main__":
    main()
