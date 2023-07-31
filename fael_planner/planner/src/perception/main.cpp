#include <stdio.h>
#include "perception/PolyDetector.h"

int main(int argc, char **argv)
{
    std::vector<PolyLine> lines = {
        { { 0.481273, 19.263916, 0.000000 }, { 2.672669, -20.676010, 0.000000 } },
        { { -2.632347, 18.215017, 0.000000 }, { 5.508471, -21.049944, 0.000000 } },
        { { -17.857944, 5.812509, 0.000000 }, { 20.700626, -5.557483, 0.000000 } },
        { { -20.295795, -2.343559, 0.000000 }, { 19.730148, 2.348336, 0.000000 } },
        { { -20.073170, -2.086306, 0.000000 }, { 20.038513, 2.731704, 0.000000 } },
        { { -23.044189, -3.178784, 0.000000 }, { 16.882099, 3.613972, 0.000000 } },
        { { -23.059996, -7.303123, 0.000000 }, { 14.749506, 7.488816, 0.000000 } },
        { { -24.727707, -3.507834, 0.000000 }, { 15.546275, 2.365544, 0.000000 } },
        { { 18.502918, -8.634955, 0.000000 }, { -20.374165, -21.012856, 0.000000 } },
        { { 9.202598, 4.523481, 0.000000 }, { -7.426190, -32.843529, 0.000000 } },
        { { 20.847351, -7.539667, 0.000000 }, { -18.186256, -20.084688, 0.000000 } },
        { { 17.327255, -5.417993, 0.000000 }, { -19.222294, -24.215343, 0.000000 } },
        { { 15.582670, -8.607574, 0.000000 }, { -23.266512, -22.325489, 0.000000 } },
        { { -1.368334, -31.398386, 0.000000 }, { -13.871298, 7.963598, 0.000000 } },
        { { -1.544255, -29.319807, 0.000000 }, { -14.729448, 9.924440, 0.000000 } },
        { { -2.539731, -27.353601, 0.000000 }, { -13.925552, 12.553957, 0.000000 } },
        { { -3.332984, -26.400829, 0.000000 }, { -15.176960, 13.477487, 0.000000 } },
        { { -4.918091, -22.149073, 0.000000 }, { -14.987379, 18.316957, 0.000000 } },
        { { -4.509918, -21.885679, 0.000000 }, { -14.484156, 18.706865, 0.000000 } },
        { { -3.773709, -22.967257, 0.000000 }, { -16.725311, 16.880781, 0.000000 } },
        { { 11.133360, 1.055563, 0.000000 }, { -29.372477, -10.047474, 0.000000 } },
        { { -27.483515, -14.038325, 0.000000 }, { 9.971365, 5.185147, 0.000000 } },
        { { -6.512562, -22.113499, 0.000000 }, { -14.757652, 19.273197, 0.000000 } },
        { { -29.544163, -5.129999, 0.000000 }, { 11.304194, 5.856430, 0.000000 } },
        { { -29.245459, -2.901008, 0.000000 }, { 12.582209, 4.042094, 0.000000 } },
        { { -26.603186, -5.194456, 0.000000 }, { 14.157106, 6.840844, 0.000000 } },
        { { -26.637554, -3.911626, 0.000000 }, { 14.771155, 6.092309, 0.000000 } },
        { { -0.640467, -20.712761, 0.000000 }, { -9.923534, 20.965946, 0.000000 } },
    };

    PolyDetector pd;
	//pd.verbose = 1;
	
    for (auto &l : lines)
        pd.AddLine(l);

    if (!pd.DetectPolygons())
    {
        logoutf("%s", "WARN: cannot detect polys!");
        return -1;
    }
	
	logoutf("nPolys:%u dissolveSteps:%u lines:%u", uint32_t(pd.polys.size()), pd.dissolveCount + 1, uint32_t(pd.lines.size()));

#if 0
    for (auto &poly : pd.polys)
    {
        for (auto &p : poly.p)
        {
            logoutf("[%u] p:{%f %f}", poly.id, p.x, p.y);
        }
    }
#endif

    return 0;
}

