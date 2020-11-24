import itertools


def readStockPrice():
    goreNormalizedProbability = {}
    with open( 'Data/StockPrices.txt' ) as f:
        for line1, line2, line3 in itertools.zip_longest( *[f] * 3 ):
            dem = line1.split()
            rep = line3.split()
            denominator = float( dem[-1] ) + float( rep[-1] )
            date = dem[0]
            normProb = float( dem[-1] ) / denominator
            goreNormalizedProbability.update( {date: normProb} )
    return goreNormalizedProbability


def main():
    goreNormProbability = readStockPrice()
    print( goreNormProbability )


if __name__ == "__main__":
    main()
