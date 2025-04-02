
using System.Reflection;
using System.Runtime.InteropServices;

namespace Regression_analysis
{
    public class RegressionEvaluator
    {
        
        //public delegate double LikelihoodFunction(IModel model, Vectors theta, Vectors[] @params);

        //public delegate Vectors dLikelihoodFunction(IModel model, Vectors theta, Vectors[] @params);

        public static Vectors? Fit(
            IModel model,
            object evolution,
            int countIteration,
            int countObservations,
            TypeDisribution errorDist,
            Vectors paramsDist,
            bool isRound = false,
            int? roundDecimals = null,
            int? seed = null
            ) 
        {

            if (countIteration <= 0 || countObservations <= 0) return null;
            Vectors Statistics = Vectors.InitVectors((1, countIteration));
            var Interval = new Vectors([-1e+5, 1e+5]);
            var generator = seed is null ? new Random() : new Random((int) seed);
            (double, double)[] intervalsObservations = new (double, double)[model.CountFacts];
            Vectors x, y, matrixX, vectorE, calcTheta;
            if (evolution is MNK mnk) {
                
            }
            else if (evolution is MMK mmk)
            {
                 
            }
            else return null;

            if (errorDist is TypeDisribution.Normal) {
                if (!NormalDistribution.CheckParamsDist(paramsDist)) return null;
                for (int i = 0; i < countIteration; i++) {
                    intervalsObservations = [];
                    double a = (double) UniformDistribution.Generate(Interval, generator);
                    double b = (double) UniformDistribution.Generate(Interval, generator);
                    if (a > b)
                        (a, b) = (b, a);
                    for (int j = 0; j < model.CountFacts; j++)
                        intervalsObservations[j] = (a, b);
                    x = LinespaceRandom.Generate((countObservations, model.CountFacts), intervalsObservations, generator);
                    matrixX = model.CreateMatrixX(x);
                    vectorE = NormalDistribution.Generate((1, countObservations), paramsDist, generator);
                    y = (matrixX & model.TrueTheta.T()).T() + vectorE;
                    calcTheta = func
                }
            }
        }
    }
}
