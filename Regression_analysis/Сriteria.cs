
using RegressionAnalysisLibrary;

namespace Regression_analysis
{
    public static class SREСriteria
    {
        private static double CalcRSS(Vectors calcTheta, Vectors matrixX, Vectors y)
        {
            var vector = y - (matrixX & calcTheta.T()).T();
            vector *= vector;
            return Vectors.Sum(vector);
        }
        private static double CalcRSSH(Vectors y)
        {
            double result = 0.0;
            double avgY = Vectors.Sum(y) / y.Size;
            for (int i = 0; i < y.Size; i++)
            {
                result += Math.Pow(y[i] - avgY, 2);
            }
            return result;
        }
        public static double SRE(IModel model, Vectors calcTheta, Vectors x, Vectors y)
        {
            double rSS = CalcRSS(calcTheta, model.CreateMatrixX(x), y);
            double rSS_H = CalcRSSH(y);
            return (rSS_H - rSS) * (y.Size - model.CountRegressor) / (rSS * (model.CountRegressor - 1));
        }
    }

    public static class SREСriteriaLR
    {
        private static double CalcRSS(Vectors calcTheta, Vectors matrixX, Vectors y)
        {
            var vector = y - (matrixX & calcTheta.T()).T();
            vector *= vector;
            return Vectors.Sum(vector);
        }
        public static double SRELR(LogLikelihoodFunction logfunction, IModel model, Vectors calcTheta, Vectors[] parameters)
        {
            // Полное логарифмическое правдоподобие
            double llFull = logfunction(model, calcTheta, parameters);

            Vectors thetaRestricterd;
            if (model.FreeMember)
            {
                thetaRestricterd = Vectors.Zeros((1, model.CountRegressor));
                thetaRestricterd[0] = calcTheta[0];
            }
            else
            {
                thetaRestricterd = Vectors.Zeros((1, model.CountRegressor));
            }

            // Ограниченное логарифмическое правдоподобие (только intercept)
            double llRestricted = logfunction(model, thetaRestricterd, parameters);

            var lR = -2 * (llFull - llRestricted);

            double rSS = CalcRSS(calcTheta, model.CreateMatrixX(parameters[1]), parameters[2]);
            return lR * (parameters[2].Size - model.CountRegressor) / (rSS * (model.CountRegressor - 1));
        }
    }

    public static class LRCriteria
    {
        public static double CalculateLR
            (
            LogLikelihoodFunction logfunction,
            IModel model,
            Vectors thetaFull,
            Vectors[] parameters
            )
        {
            // Полное логарифмическое правдоподобие
            double llFull = logfunction(model, thetaFull, parameters);

            Vectors thetaRestricterd;
            if (model.FreeMember) {
                thetaRestricterd = Vectors.Zeros((1, model.CountRegressor));
                thetaRestricterd[0] = thetaFull[0];
            }
            else
            {
                thetaRestricterd = Vectors.Zeros((1, model.CountRegressor));
            }

            // Ограниченное логарифмическое правдоподобие (только intercept)
            double llRestricted = logfunction(model, thetaRestricterd, parameters);

            var lR = -2 * (llFull - llRestricted);
            // LR-статистика
            return ((parameters[2].Size - model.CountRegressor) / (model.CountRegressor - 1)) * double.ExpM1(lR / parameters[2].Size);
        }

    }
}