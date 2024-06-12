#pragma once
#include <functional>
#include <algorithm>
#include <math.h>
namespace rbl
{
    typedef std::function<double(double)> func;
    struct BracketOutput
    {
        bool result;
        double lower_bound;
        double upper_bound;
    };

    void bracket_minimum(func f, BracketOutput &out, double x = 0, double s = 1e-2, double k = 2.0)
    {
        double a = x;
        double ya = f(x);
        double b = a + s;
        double yb = f(a + s);
        double c = 0, yc = 0;
        double s_in = s;
        double k_in = k;

        if (yb > ya)
        {
            std::swap(a, b);
            std::swap(ya, yb);
            s_in = -s_in;
        }

        for (int i = 0; i < 1e7; i++)
        {
            c = b + s;
            yc = f(b + s);
            if (yc > yb)
            {
                out.result = true;
                out.lower_bound = a < c ? a : c;
                out.upper_bound = a < c ? c : a;
                return;
            }
        }
        out.result = false;
        return;
    }

    void fibonacci_search(func f, BracketOutput &out, double a, double b, int n, double espi = 0.01)
    {
        double s = (1 - sqrt(5)) / (1 + sqrt(5));
        double rho = (1 - pow(s, n)) / (1.61803 * (1 - pow(s, (n + 1))));
        double d = rho * b + (1 - rho) * a;
        double yd = f(d);
        double c = 0, yc = 0;
        for (int i = 1; i < n; i++)
        {
            if (i == n - 1)
            {
                c = espi * a + (1 - espi) * d;
            }
            else
            {
                c = rho * a + (1 - rho) * b;
            }
            yc = f(c);
            if (yc < yd)
            {
                b = d;
                d = c;
                yd = yc;
            }
            else
            {
                a = b;
                b = c;
            }
            rho = (1 - pow(s, n - i)) / (1.61803 * (1 - pow(s, (n - i + 1))));
        }
        out.result = true;
        out.lower_bound = a < b ? a : b;
        out.upper_bound = a < b ? b : a;
    }

    void golden_section_search(func f, BracketOutput &out, double a, double b, int n)
    {
        double phi = 1.61803;
        double rho = phi - 1;
        double d = rho * b + (1 - rho) * a;
        double yd = f(d);
        double c = 0, yc = 0;
        for (int i = 1; i < n; i++)
        {
            c = rho * a + (1 - rho) * b;
            yc = f(c);
            if (yc < yd)
            {
                b = d;
                d = c;
                yd = yc;
            }
            else
            {
                a = b;
                b = c;
            }
        }
        out.result = true;
        out.lower_bound = a < b ? a : b;
        out.upper_bound = a < b ? b : a;
    }

    void quadratic_fit_search(func f, BracketOutput& out, double a, double b, double c, int n)
    {
        double ya = f(a);
        double yb = f(b);
        double yc = f(c);
        double x =0, yx = 0;
        for(int i=1;i<n-2;i++)
        {
            x = 0.5*(ya*(pow(b,2) - pow(c,2)) + yb*(pow(c,2)-pow(a,2))+yc*(pow(a,2)-pow(b,2))) /
                (ya * (b-c) + yb*(c-a) + yc*(a-b));
            yx = f(x);
            if(x>b)
            {
                if(yx>yb)
                {
                    c = x;
                    yc = yx;
                }
                else
                {
                    a = b;
                    ya = yb;
                    b = x;
                    yb = yx;
                }
            }
            else if (x < b)
            {
                if(yx > yb)
                {
                    a = x;
                    ya = yx;
                }
                else
                {
                    c = b;
                    yc = yb;
                    b = x;
                    yb = yx;
                }
            }
            
        }
        out.result = true;
        out.lower_bound = a;
        out.upper_bound = c;
    }

    int sign(double x)
    {
        if(x > 0) return 1;
        if(x < 0) return -1;
        return 0;
    }

    void bisection(func f, BracketOutput& out, double a, double b, double epsi)
    {
        if(a>b)
        {
            std::swap(a,b);
        }
        double ya = f(a);
        double yb = f(b);
        if(ya == 0)
        {
            b = a;
        }
        if(yb == 0)
        {
            a = b;
        }
        double x = 0, y = 0;
        while(b -a > epsi)
        {
            x = (a+b)/2;
            y = f(x);
            if( y == 0)
            {
                a = x;
                b = x;
            }
            else if(sign(y) == sign(ya))
            {
                a = x;
            }
            else
            {
                b = x;
            }

        }
        out.result = true;
        out.lower_bound = a;
        out.upper_bound = b;
    }

    void bracket_sign_change(func f, BracketOutput& out,double a, double b, double k = 2.0)
    {
        if(a>b)
        {
            std::swap(a,b);
        }
        double center = (a+b)/2.0;
        double half_width = (b-a)/2.0;
        while (f(a) * f(b) > 0)
        {
            half_width *=k;
            a = center - half_width;
            b = center + half_width;
        }
        
        out.result = true;
        out.lower_bound = a;
        out.upper_bound = b;
    }
} // namespace rbl
