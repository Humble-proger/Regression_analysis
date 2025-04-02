using System.Globalization;

namespace WinFormsApp1
{
    
    
    
    internal static class Program
    {
        /// <summary>
        ///  The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            // To customize application configuration such as set high DPI settings or default font,
            // see https://aka.ms/applicationconfiguration.
            var points = ParsePointsFromFile("D:\\Program\\Budancev\\��\\��\\uniform.dat");

            ApplicationConfiguration.Initialize();
            Application.Run(new Form1(points));
        }

        public static PointF[] ParsePointsFromFile(string filePath)
        {
            string[] lines = File.ReadAllLines(filePath);

            // ���������� ������ ������ (���������)
            if (lines.Length < 2)
                throw new ArgumentException("���� ������ ��������� ��� ������� ��� ������.");

            // ��������� ������ ������, ����� ������ ���������� �����
            string[] secondLineParts = lines[1].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (secondLineParts.Length < 2 || !int.TryParse(secondLineParts[1], out int pointCount))
                throw new ArgumentException("�� ������ ������ ������ ���� ������� ���������� �����.");

            // ������ ��� ����� �� ����� (������� �� ������ ������)
            var numbers = lines
                .Skip(1) // ���������� ������ ������
                .SelectMany(line => line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries))
                .Select(numStr => double.Parse(numStr, CultureInfo.InvariantCulture))
                .ToArray();

            // ���������, ��� ����� ���������� ��� ������������ �����
            pointCount /= 2;
            if (numbers.Length < pointCount * 2)
                throw new ArgumentException($"��������� {pointCount * 2} �����, �� ������� ������ {numbers.Length}.");

            // ������ ������ �����
            var points = new PointF[pointCount];
            for (int i = 0; i < pointCount; i++)
            {
                float x = (float) numbers[i * 2];
                float y = (float) numbers[i * 2 + 1];
                points[i] = new PointF(x, y);
            }

            return points;
        }
    }
}