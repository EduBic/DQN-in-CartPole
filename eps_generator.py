
import csv
import math

def main():

    num_step = 10000 # steps of the agent
    max_eps = 1
    min_eps = 0.01
    mLambda = 0.001


    fileCsvPath = "eps-gen-Min" + str(min_eps) + "-Max" + str(max_eps) + "-Lambda" + str(mLambda) + ".csv"

    with open(fileCsvPath, 'w', newline='') as csvfile:
        fieldnames = ['step', 'epsilon']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        const_ = min_eps + (max_eps - min_eps)

        for step in range(num_step):
            writer.writerow({
                fieldnames[0]: step + 1,
                fieldnames[1]: const_ * math.exp(- mLambda * step)   # epsilon
            })




if __name__ == "__main__":
    main()