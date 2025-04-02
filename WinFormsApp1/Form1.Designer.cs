using Microsoft.VisualBasic.Logging;

namespace WinFormsApp1
{
    partial class Form1
    {
        private PointF[] points;
        private float centerX = 0;
        private float centerY = 0;
        private float scale = 1.0f;
        private Point? dragStart = null;
        private PointF? centerStart = null;

        public Form1(PointF[] points)
        {
            this.points = points;
            InitializeComponents();
            this.DoubleBuffered = true;
            AutoCenterAndScale(); // Автоматическая настройка при запуске
        }

        private void InitializeComponents()
        {
            this.Text = "Интерактивный график";
            this.ClientSize = new Size(1000, 700);

            // Панель управления
            var panel = new Panel { Dock = DockStyle.Right, Width = 200, BackColor = Color.LightGray };

            // Находим диапазон значений для ограничения NumericUpDown
            float minVal = Math.Min(points.Min(p => p.X), points.Min(p => p.Y));
            float maxVal = Math.Max(points.Max(p => p.X), points.Max(p => p.Y));
            float range = Math.Max(Math.Abs(minVal), Math.Abs(maxVal)) * 2;

            // Элементы управления
            var lblCenterX = new Label { Text = "Центр X:", Top = 20, Left = 10 };
            var numCenterX = new NumericUpDown
            {
                Top = 40,
                Left = 10,
                Width = 150,
                DecimalPlaces = 2,
                Minimum = (decimal) (-range),
                Maximum = (decimal) range,
                Increment = 0.1m
            };

            var lblCenterY = new Label { Text = "Центр Y:", Top = 70, Left = 10 };
            var numCenterY = new NumericUpDown
            {
                Top = 90,
                Left = 10,
                Width = 150,
                DecimalPlaces = 2,
                Minimum = (decimal) (-range),
                Maximum = (decimal) range,
                Increment = 0.1m
            };

            var lblScale = new Label { Text = "Масштаб:", Top = 120, Left = 10 };
            var trackScale = new TrackBar
            {
                Top = 140,
                Left = 10,
                Width = 150,
                Minimum = 1,
                Maximum = 1000,
                SmallChange = 10,
                LargeChange = 100,
                TickFrequency = 100
            };

            // Кнопки управления
            var btnAuto = new Button { Text = "Автонастройка", Top = 180, Left = 10, Width = 150 };
            var btnZoomIn = new Button { Text = "Увеличить (+)", Top = 220, Left = 10, Width = 70 };
            var btnZoomOut = new Button { Text = "Уменьшить (-)", Top = 220, Left = 90, Width = 70 };

            // Добавление элементов на панель
            panel.Controls.AddRange(new Control[] { lblCenterX, numCenterY, lblCenterY, numCenterX,
                                            lblScale, trackScale, btnAuto, btnZoomIn, btnZoomOut });

            this.Controls.Add(panel);

            // Инициализация значений
            numCenterX.Value = (decimal) centerX;
            numCenterY.Value = (decimal) centerY;
            trackScale.Value = (int) (scale * 100);

            // Обработчики событий
            numCenterX.ValueChanged += (s, e) => { centerX = (float) numCenterX.Value; this.Invalidate(); };
            numCenterY.ValueChanged += (s, e) => { centerY = (float) numCenterY.Value; this.Invalidate(); };
            trackScale.ValueChanged += (s, e) => { scale = trackScale.Value / 100f; this.Invalidate(); };

            btnAuto.Click += (s, e) =>
            {
                AutoCenterAndScale();
                numCenterX.Value = (decimal) centerX;
                numCenterY.Value = (decimal) centerY;
                trackScale.Value = (int) (scale * 100);
            };

            btnZoomIn.Click += (s, e) => { trackScale.Value = Math.Min(trackScale.Maximum, trackScale.Value + 50); };
            btnZoomOut.Click += (s, e) => { trackScale.Value = Math.Max(trackScale.Minimum, trackScale.Value - 50); };

            // Обработчики мыши
            this.MouseDown += (s, e) =>
            {
                if (e.Button == MouseButtons.Left && e.X < this.ClientSize.Width - 200)
                {
                    dragStart = e.Location;
                    centerStart = new PointF(centerX, centerY);
                }
            };

            this.MouseMove += (s, e) =>
            {
                if (dragStart.HasValue && centerStart.HasValue)
                {
                    var dx = (e.X - dragStart.Value.X) / scale;
                    var dy = (e.Y - dragStart.Value.Y) / scale;
                    centerX = centerStart.Value.X - dx;
                    centerY = centerStart.Value.Y + dy;
                    numCenterX.Value = (decimal) centerX;
                    numCenterY.Value = (decimal) centerY;
                    this.Invalidate();
                }
            };

            this.MouseUp += (s, e) => { dragStart = null; centerStart = null; };

            this.MouseWheel += (s, e) =>
            {
                trackScale.Value = Math.Max(trackScale.Minimum,
                    Math.Min(trackScale.Maximum, trackScale.Value + (e.Delta > 0 ? 20 : -20)));
            };
        }

        private void AutoCenterAndScale()
        {
            centerX = points.Average(p => p.X);
            centerY = points.Average(p => p.Y);

            var minX = points.Min(p => p.X);
            var maxX = points.Max(p => p.X);
            var minY = points.Min(p => p.Y);
            var maxY = points.Max(p => p.Y);

            var plotWidth = this.ClientSize.Width - 220;
            var plotHeight = this.ClientSize.Height - 40;

            var xScale = plotWidth / (maxX - minX) * 0.9f;
            var yScale = plotHeight / (maxY - minY) * 0.9f;

            scale = Math.Min(xScale, yScale);
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            var g = e.Graphics;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            g.Clear(Color.White);

            // Размеры области рисования
            int margin = 20;
            int plotWidth = this.ClientSize.Width - 220;
            int plotHeight = this.ClientSize.Height - 40;
            int centerScreenX = margin + plotWidth / 2;
            int centerScreenY = margin + plotHeight / 2;

            // Функция преобразования координат
            PointF Transform(PointF point)
            {
                float x = centerScreenX + (point.X - centerX) * scale;
                float y = centerScreenY - (point.Y - centerY) * scale;
                return new PointF(x, y);
            }

            // Рисуем оси
            var axisPen = new Pen(Color.Black, 1.5f);
            g.DrawLine(axisPen, margin, centerScreenY, margin + plotWidth, centerScreenY);
            g.DrawLine(axisPen, centerScreenX, margin, centerScreenX, margin + plotHeight);

            // Рисуем сетку
            var gridPen = new Pen(Color.LightGray, 0.5f);
            DrawGrid(g, gridPen, centerScreenX, centerScreenY, plotWidth, plotHeight);

            // Рисуем точки
            var pointBrush = new SolidBrush(Color.Red);
            foreach (var point in points)
            {
                var transformed = Transform(point);
                if (transformed.X >= margin && transformed.X <= margin + plotWidth &&
                    transformed.Y >= margin && transformed.Y <= margin + plotHeight)
                {
                    g.FillEllipse(pointBrush, transformed.X - 1, transformed.Y - 1, 2, 2);
                }
            }

            // Подписи
            var font = new Font("Arial", 10);
            g.DrawString($"Центр: ({centerX:F2}, {centerY:F2})", font, Brushes.Black, 10, 10);
            g.DrawString($"Масштаб: {scale:F2}x", font, Brushes.Black, 10, 30);
        }

        private void DrawGrid(Graphics g, Pen pen, int centerX, int centerY, int width, int height)
        {
            // Вертикальные линии сетки
            float stepX = 50 / scale; // Шаг сетки в единицах данных
            for (float x = centerX - stepX; x > 20; x -= stepX)
            {
                float dataX = centerX - (x - centerX) / scale;
                g.DrawLine(pen, x, 20, x, 20 + height);
            }
            for (float x = centerX + stepX; x < 20 + width; x += stepX)
            {
                g.DrawLine(pen, x, 20, x, 20 + height);
            }

            // Горизонтальные линии сетки
            float stepY = 50 / scale;
            for (float y = centerY - stepY; y > 20; y -= stepY)
            {
                g.DrawLine(pen, 20, y, 20 + width, y);
            }
            for (float y = centerY + stepY; y < 20 + height; y += stepY)
            {
                g.DrawLine(pen, 20, y, 20 + width, y);
            }
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            SuspendLayout();
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(921, 500);
            Name = "Form1";
            Text = "Form1";
            ResumeLayout(false);
        }

        #endregion
    }
}
