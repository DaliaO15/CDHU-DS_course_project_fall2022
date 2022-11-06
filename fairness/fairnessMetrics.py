class FairnessMetrics():
    def equalizedOdds(outcomes):
        return (outcomes.tp/(outcomes.tp + outcomes.tn), outcomes.fp/(outcomes.fn + outcomes.fp))

    def equalizedOddsDiff(outcomes1, outcomes2):
        eO1 = FairnessMetrics.equalizedOdds(outcomes1)
        eO2 = FairnessMetrics.equalizedOdds(outcomes2)
        return (abs(eO2[0]-eO1[0]), abs(eO2[1]-eO1[1]))


    def statisticalParity(outcomes):
        return (outcomes.tp + outcomes.fp)/(outcomes.tp + outcomes.fn + outcomes.tn + outcomes.fp)

    def statisticalParityDiff(outcomes1, outcomes2):
        return abs(FairnessMetrics.statisticalParity(outcomes2) - FairnessMetrics.statisticalParity(outcomes1))


    def predictiveParity(outcomes):
        return outcomes.tp/(outcomes.tp + outcomes.fp)

    def predictiveParityDiff(outcomes1, outcomes2):
        return abs(FairnessMetrics.predictiveParity(outcomes2) - FairnessMetrics.predictiveParity(outcomes1))
