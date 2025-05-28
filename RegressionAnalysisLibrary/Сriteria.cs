namespace RegressionAnalysisLibrary
{
    public enum Criteria
    {
        SRE,
        LR
    }

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
            var result = 0.0;
            var avgY = Vectors.Sum(y) / y.Size;
            for (var i = 0; i < y.Size; i++)
                result += Math.Pow(y[i] - avgY, 2);
            return result;
        }
        public static double SRE(IModel model, Vectors calcTheta, Vectors x, Vectors y)
        {
            var rSS = CalcRSS(calcTheta, model.CreateMatrixX(x), y);
            var rSS_H = CalcRSSH(y);
            return (rSS_H - rSS) * (y.Size - model.CountRegressor) / (rSS * (model.CountRegressor - 1));
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
            var llFull = logfunction(model, thetaFull, parameters);

            Vectors thetaRestricterd;
            if (model.FreeMember)
            {
                thetaRestricterd = Vectors.Zeros((1, model.CountRegressor));
                thetaRestricterd[0] = thetaFull[0];
            }
            else
                thetaRestricterd = Vectors.Zeros((1, model.CountRegressor));

            // Ограниченное логарифмическое правдоподобие (только intercept)
            var llRestricted = logfunction(model, thetaRestricterd, parameters);

            // LR-статистика
            return 2 * (llRestricted - llFull);
        }

    }
}