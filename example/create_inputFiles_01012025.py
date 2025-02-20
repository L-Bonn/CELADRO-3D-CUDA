import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# ---------------------------
# Utility functions
# ---------------------------
def fRand(fMin, fMax):
    """Return a random float between fMin and fMax."""
    return random.uniform(fMin, fMax)

def plot_lattice(xcf, ycf, Lx, Ly):
    """
    Plot lattice points (xcf, ycf) with boundaries defined by Lx and Ly.
    
    Parameters:
        xcf (list or array): The x-coordinates of the lattice points.
        ycf (list or array): The y-coordinates of the lattice points.
        Lx (float): The maximum x-boundary.
        Ly (float): The maximum y-boundary.
    """
    # Create a new figure and axis.
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the lattice points.
    ax.scatter(xcf, ycf, s=50, color='blue', label='Lattice Points')
    
    # Set the axis limits.
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    # Optionally, add a rectangle to show the boundary explicitly.
    boundary_rect = Rectangle((0, 0), Lx, Ly,
                              linewidth=2, edgecolor='red',
                              facecolor='none', label='Boundary')
    ax.add_patch(boundary_rect)
    
    # Add labels and title.
    ax.set_xlabel('x-coordinate')
    ax.set_ylabel('y-coordinate')
    ax.set_title('Lattice Plot with Boundaries')
    
    # Add a legend.
    ax.legend()
    
    # Adjust the layout to prevent clipping of labels.
    plt.tight_layout()
    
    # Display the plot.
    #plt.show()
    plt.savefig("check_lattice.png", format='png', dpi=300)
    

# ---------------------------
# File output functions
# ---------------------------

def create_triangular_lattice(ncells, lbox, zcoor):
    """
    Create a triangular lattice of cell centers with a given margin (lbox) from the boundaries.
    
    Parameters:
        ncells (int): Total number of cells (must be a perfect square).
        lbox (float): Used both as the spacing between cells and as the margin from each boundary.
        zcoor (float): The fixed z-coordinate for all cell centers.
    
    Returns:
        xcf (list of float): x-coordinates of the cell centers.
        ycf (list of float): y-coordinates of the cell centers.
        zcf (list of float): z-coordinates of the cell centers (all equal to zcoor).
        Lx (float): Upper bound in the x-direction.
        Ly (float): Upper bound in the y-direction.
        
    The function arranges the cells in a triangular (or hexagonal) lattice:
      - For even-numbered rows, the x-coordinate starts at lbox.
      - For odd-numbered rows, an offset of lbox/2 is added.
      - The vertical spacing between rows is lbox * (sqrt(3)/2).
    """
    N = int(math.sqrt(ncells))
    if N * N != ncells:
        raise ValueError("ncells must be a perfect square for a triangular lattice.")
    
    xcf = []
    ycf = []
    zcf = []
    
    # The vertical spacing between rows.
    dy = lbox * (math.sqrt(3) / 2)
    
    # Loop over rows and columns.
    for j in range(N):
        for i in range(N):
            # For even rows (j even), no horizontal offset.
            # For odd rows, add an offset of lbox/2.
            if j % 2 == 0:
                x = lbox + i * lbox
            else:
                x = lbox + (i + 0.5) * lbox
            y = lbox + j * dy
            
            xcf.append(x)
            ycf.append(y)
            zcf.append(zcoor)
    
    # Determine the domain boundaries.
    # For x: even rows go up to lbox + (N-1)*lbox = N*lbox.
    #         odd rows go up to lbox + (N-1 + 0.5)*lbox = (N+0.5)*lbox.
    # Use the larger value then add the margin lbox.
    max_x = (N + 0.5) * lbox if N > 1 else lbox
    Lx = max_x + lbox
    
    # For y: The top of the last row is at y = lbox + (N-1)*dy.
    # Adding the margin gives:
    Ly = (lbox + (N - 1) * dy) + lbox
    
    return xcf, ycf, zcf, Lx, Ly

def create_square_lattice(ncells, lbox, zcoor):
    """
    Create a square lattice of cell centers with a given margin (lbox) from the boundaries.
    
    Parameters:
        ncells (int): Total number of cells (must be a perfect square).
        lbox (float): Minimum distance from any cell center to each side.
                      This value is also used as the spacing between adjacent cell centers.
    
    Returns:
        xcf (list of float): x-coordinates of the cell centers.
        ycf (list of float): y-coordinates of the cell centers.
        Lx (float): Upper bound in the x-direction.
        Ly (float): Upper bound in the y-direction.
        
    Example:
        For ncells = 16 and lbox = 10:
          - The lattice will have 4 rows and 4 columns.
          - The cell centers will be at (10,10), (20,10), ..., (40,40).
          - The domain bounds will be Lx = Ly = 50, giving a 10-unit margin on each side.
    """
    N = int(math.sqrt(ncells))
    if N * N != ncells:
        raise ValueError("ncells must be a perfect square for a square lattice.")
    
    xcf = []
    ycf = []
    zcf = []
    
    # Loop over rows and columns.
    for j in range(N):
        for i in range(N):
            # Place the cell center so that the first center is at (lbox, lbox),
            # the next at (2*lbox, lbox), etc.
            x = lbox + i * lbox
            y = lbox + j * lbox
            xcf.append(x)
            ycf.append(y)
            zcf.append(zcoor)
    
    # The upper boundary is lbox beyond the last cell center.
    Lx = (N + 1) * lbox
    Ly = (N + 1) * lbox
    
    return xcf, ycf, zcf, Lx, Ly

def write_simCard(config, config_subAdh, gamma, omega_cc, omega_cs, alpha,
                    nsteps, ninfo, Lx, Ly, Lz, nsubsteps, bc, margin, relax_time,
                    nphases, mu, lambda_, kappa_cs, R0, xi, wallThich, kappa_cc,
                    SPOL, DPOL, JPOL, KPOL, kij, zetaS, zetaQ, SNEM, KNEM, JNEM, WNEM,
                    count):
    """Write the simulation card file with the given parameters."""
    filename = f"simCard_{count}.dat"
    with open(filename, "w") as f:
        f.write("# Sample runcard\n")
        f.write(f"config = {config}\n")
        f.write(f"nsteps = {nsteps}\n")
        f.write(f"ninfo = {ninfo}\n")
        f.write(f"LX = {Lx}\n")
        f.write(f"LY = {Ly}\n")
        f.write(f"LZ = {Lz}\n")
        f.write(f"nsubsteps = {nsubsteps}\n")
        f.write(f"bc = {bc}\n")
        f.write(f"margin = {margin}\n")
        f.write(f"relax-time = {relax_time}\n")
        f.write(f"substrate-adhesion = {config_subAdh}\n")
        f.write(f"nphases = {nphases}\n")
        f.write(f"gamma = {gamma}\n")
        f.write(f"mu = {mu}\n")
        f.write(f"lambda = {lambda_}\n")
        f.write(f"kappa_cc = {kappa_cc}\n")
        f.write(f"R = {R0}\n")
        f.write(f"xi = {xi}\n")
        f.write(f"omega_cc = {omega_cc}\n")
        f.write(f"wall-thickness = {wallThich}\n")
        f.write(f"kappa_cs= {kappa_cs}\n")
        f.write(f"omega_cs = {omega_cs}\n")
        f.write(f"alpha = {alpha}\n")
        f.write(f"kij = {kij}\n")
        f.write(f"S-pol = {SPOL}\n")
        f.write(f"D-pol = {DPOL}\n")
        f.write(f"J-pol = {JPOL}\n")
        f.write(f"K-pol = {KPOL}\n")
        f.write(f"zetaS = {zetaS}\n")
        f.write(f"zetaQ = {zetaQ}\n")
        f.write(f"S-nem = {SNEM}\n")
        f.write(f"K-nem = {KNEM}\n")
        f.write(f"J-nem = {JNEM}\n")
        f.write(f"W-nem = {WNEM}\n")


def write_posfile_mix_perc(nparticles, xcf, ycf, zcf, zcoor, na,
                           zetas1, zetas2, zetaQ1, zetaQ2,
                           gam1, gam2, omega1, omega2, omega_wall1, omega_wall2,
                           kappa_cc, mu, alpha, xi, R, count):
    """Write a position file for a mixed percentage of two types."""
    filename = f"input_str_{count}.dat"
    with open(filename, "w") as f:
        for j in range(nparticles):
            xc = xcf[j]
            yc = ycf[j]
            # Always use zcoor as in the original code
            if j < na:
                # Type 1 cell
                f.write(f"1 {xc} {yc} {zcoor} {zetas1} {zetaQ1} {gam1} {omega1} {omega_wall1} "
                        f"{kappa_cc} {mu} {alpha} {xi} {R}\n")
            else:
                # Type 2 cell
                f.write(f"2 {xc} {yc} {zcoor} {zetas2} {zetaQ2} {gam2} {omega2} {omega_wall2} "
                        f"{kappa_cc} {mu} {alpha} {xi} {R}\n")


def write_summary(count, gam1, gam2, zetas1, zetas2,
                  omega_cc1, omega_cc2, omega_cs1, omega_cs2, alpha, xi, zetaQ):
    """Append a summary of simulation parameters to a summary file."""
    filename = "simulation_parameter_summary.dat"
    with open(filename, "a") as f:
        f.write(f"{count} {gam1} {gam2} {zetas1} {zetas2} {omega_cc1} {omega_cc2} "
                f"{omega_cs1} {omega_cs2} {alpha} {xi} {zetaQ}\n")


def Export(filename, tmp):
    """Export a list of numbers to a file, one per line."""
    with open(filename, "w") as f:
        for val in tmp:
            f.write(f"{val}\n")


# ---------------------------
# Lattice Initialization Functions
# ---------------------------
def disorder_initial(l, nx, ny, nz):
    """
    Create a disordered lattice.
    Returns three lists (xcf, ycf, zcf) of coordinates.
    """
    xcf, ycf, zcf = [], [], []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                xc = round(l * x + l/2)
                yc = round(l * y + l/2)
                zc = round(l * z + l/2)
                xcf.append(xc)
                ycf.append(yc)
                zcf.append(zc)
    return xcf, ycf, zcf


def init_square_lattice(l, nx, ny, nz):
    """
    Initialize a square lattice.
    Returns three lists (xcf, ycf, zcf) of coordinates.
    """
    # This implementation is essentially the same as disorder_initial.
    return disorder_initial(l, nx, ny, nz)


def init_triangular_lattice(nx, ny, domainWidth, domainHeight, zcoor, R0):
    """
    Initialize a triangular lattice.
    Returns three lists (xcf, ycf, zcf) of coordinates.
    """
    dx = domainWidth / nx
    dy = domainHeight / ny
    dx2 = dx * 0.75  # Horizontal shift for alternate rows
    xcf, ycf, zcf = [], [], []
    for j in range(ny):
        for i in range(nx):
            x = i * dx
            if j % 2 == 1:
                x += dx2
            y = j * dy
            xcf.append(x)
            ycf.append(y + R0)  # As in C++: y + R0
            zcf.append(zcoor)
    print("Number of points:", len(xcf))
    return xcf, ycf, zcf


def disorder_mc(d_max, xcf, ycf):
    """
    Apply a Monte Carlo displacement to the lattice positions.
    Modifies the coordinate lists in place.
    """
    npart = len(xcf)
    for i in range(npart):
        exp1 = random.randint(0, 1)
        exp2 = random.randint(0, 1)
        exp3 = random.randint(0, 1)
        p1 = fRand(0, d_max)
        p2 = fRand(0, d_max)
        p3 = fRand(0, d_max)
        xcf[i] = xcf[i] + ((-1) ** exp1) * p1
        ycf[i] = ycf[i] + ((-1) ** exp2) * p2
        #zcf[i] = zcf[i] + ((-1) ** exp3) * p3


def write_lattice(filename, nparticles, xcf, ycf, zcf):
    """
    Write the lattice positions to a file.
    Note: zcoor is used for every line, as in the original C++ code.
    """
    with open(filename, "w") as f:
        for j in range(nparticles):
            f.write(f"{xcf[j]} {ycf[j]} {zcf[j]}\n")


# ---------------------------
# Other Utility Functions
# ---------------------------
def find_min(vect):
    """Return the smallest element in the list."""
    return min(vect)


def compute_all_dist(index, ix, iy, iz, txc, tyc, tzc):
    """
    Compute the minimum distance from point (ix,iy,iz) to all points in the list,
    except for the point at position 'index'.
    """
    distances = []
    for j in range(len(txc)):
        if j != index:
            d = math.sqrt((txc[j] - ix)**2 + (tyc[j] - iy)**2 + (tzc[j] - iz)**2)
            distances.append(d)
    return min(distances) if distances else None


# ---------------------------
# Main routine
# ---------------------------
def main():
    # Simulation parameters (translated from C++ variables)
    nsteps = 3000
    ninfo = 10
    #Lx = 64.0
    #Ly = 64.0
    Lz = 40.0
    ncells = 16
    R0 = 8.0
    #R0 = Lx / 2.0 / math.sqrt(ncells)
    #R0 = R0  # Overriding R0 as in the original code
    lbox = 2. * R0

    '''
    area_a = 0.5
    area_b = 1.0 - area_a
    ncel_a = round(ncells * area_a)
    ncel_b = int(ncells - ncel_a)
    ''' 
    nsubsteps = 10
    bc = 2
    margin = 18
    relax_time = 100

    gamma = 0.008
    omega_cc = 0.0008
    omega_cs = 0.002
    alpha = 0.05
    xi = 1.0
    zetaS = 0.0
    zetaQ = 0.0

    mu = 45.0
    lambda_ = 3.0  # 'lambda' is a reserved word in Python
    kappa_cc = 0.5

    wall_thickness = 7.0
    zcoor = wall_thickness + R0/2.;
    kappa_cs = 0.15
    SPOL = 1.0
    SNEM = 0.0
    KNEM = 0.0
    JNEM = 0.0
    WNEM = 0.0

    JPOL = 0.001
    KPOL = 0.001
    DPOL = 0.001
    kij = 1.0

    d_max_mc = 1.0

    #npx = round(Lx / lbox)
    #npy = round(Ly / lbox)
    #npz = 1
    nphases = int(ncells)

    print(f"Number of cells are: {nphases} with R0: {R0}")
    #print(f"Percent of type a cells: {ncel_a} and type b cells: {ncel_b}")

    # Initialize coordinates.
    # In the original code one option was a square lattice followed by a disorder MC move.
    # Here we use the triangular lattice initialization.
    # xcf, ycf, zcf = init_triangular_lattice(4, 4, 64, 64, zcoor, R0)
    # If desired, one can also call:
    # xcf, ycf, zcf = init_square_lattice(lbox, nphases, 1, 1)
    xcf, ycf, zcf, Lx, Ly = create_square_lattice(ncells, lbox, zcoor)
    Lx = int(Lx)
    Ly = int(Ly)
    Lz = int(Lz)
    disorder_mc(d_max_mc, xcf, ycf)
    write_lattice("input_str.dat", len(xcf), xcf, ycf, zcf)
    plot_lattice(xcf, ycf, Lx, Ly)
    config = "input const"
    config_subAdh = "const"

    # Define parameter arrays (each with one element in this example)
    zetaS_mix_a = [0.0]  # wt 
    zetaS_mix_b = [0.0]  # ecad 
    gamma_mix_a = [0.008]  # ecad ko: higher elasticity, lower cell-cell, lower cell-substrate
    gamma_mix_b = [0.008]
    omega_cc_mix_a = [0.0008]  # wt 
    omega_cc_mix_b = [0.0008]  # ecad 
    omega_cs_mix_a = [0.0020]  # wt 
    omega_cs_mix_b = [0.0020]  # ecad 
    zetaQ_mix_a = [0.0]  # wt 
    zetaQ_mix_b = [0.0]  # ecad 
    alpha_mix = [alpha]
    xi_mix = [xi]
    kappa_cc_mix = [kappa_cc]
    mu_mix = [mu]
    R0_mix = [R0]

    count = 1

    # Nested loops over parameter combinations.
    # Since each array has one element, only one iteration occurs.
    for i in range(len(gamma_mix_a)):
        for j in range(len(omega_cc_mix_a)):
            for k in range(len(omega_cs_mix_a)):
                for m in range(len(alpha_mix)):
                    for ii in range(len(zetaS_mix_a)):
                        for jj in range(len(zetaQ_mix_a)):
                            for kk in range(len(xi_mix)):
                                for mm in range(len(kappa_cc_mix)):
                                    for ll in range(len(mu_mix)):
                                        for hh in range(len(R0_mix)):
                                            GAMMA_A = gamma_mix_a[i]
                                            GAMMA_B = gamma_mix_b[i]
                                            
                                            ZETAS_A = zetaS_mix_a[ii]
                                            ZETAS_B = zetaS_mix_b[ii]
                                            
                                            omega_cc_A = omega_cc_mix_a[j]
                                            omega_cc_B = omega_cc_mix_b[j]
                                            
                                            omega_cs_A = omega_cs_mix_a[k]
                                            omega_cs_B = omega_cs_mix_b[k]
                                            
                                            ZETAQ_A = zetaQ_mix_a[jj]
                                            ZETAQ_B = zetaQ_mix_b[jj]
                                            
                                            ALPHA = alpha_mix[m]
                                            XI = xi_mix[kk]
                                            kappa_cc = kappa_cc_mix[mm]
                                            MU = mu_mix[ll]
                                            R0 = R0_mix[hh]

                                            # Write position file (input_str_count.dat)
                                            '''
                                            write_posfile_mix_perc(
                                                nparticles=int(ncells),
                                                xcf=xcf,
                                                ycf=ycf,
                                                zcf=zcf,
                                                zcoor=zcoor,
                                                na=ncel_a,
                                                zetas1=ZETAS_A,
                                                zetas2=ZETAS_B,
                                                zetaQ1=ZETAQ_A,
                                                zetaQ2=ZETAQ_B,
                                                gam1=GAMMA_A,
                                                gam2=GAMMA_B,
                                                omega1=omega_cc_A,
                                                omega2=omega_cc_B,
                                                omega_wall1=omega_cs_A,
                                                omega_wall2=omega_cs_B,
                                                kappa_cc=kappa_cc,
                                                mu=MU,
                                                alpha=ALPHA,
                                                xi=XI,
                                                R=R0,
                                                count=count
                                            )
                                            '''

                                            # Write simulation card (simCard_count.dat)
                                            write_simCard(
                                                config=config,
                                                config_subAdh=config_subAdh,
                                                gamma=gamma,
                                                omega_cc=omega_cc,
                                                omega_cs=omega_cs,
                                                alpha=alpha,
                                                nsteps=nsteps,
                                                ninfo=ninfo,
                                                Lx=Lx,
                                                Ly=Ly,
                                                Lz=Lz,
                                                nsubsteps=nsubsteps,
                                                bc=bc,
                                                margin=margin,
                                                relax_time=relax_time,
                                                nphases=int(ncells),
                                                mu=mu,
                                                lambda_=lambda_,
                                                kappa_cc=kappa_cc,
                                                R0=R0,
                                                xi=xi,
                                                wallThich=wall_thickness,
                                                kappa_cs = kappa_cs,
                                                SPOL=SPOL,
                                                DPOL=DPOL,
                                                JPOL=JPOL,
                                                KPOL=KPOL,
                                                kij=kij,
                                                zetaS=zetaS,
                                                zetaQ=zetaQ,
                                                SNEM=SNEM,
                                                KNEM=KNEM,
                                                JNEM=JNEM,
                                                WNEM=WNEM,
                                                count=count
                                            )

                                            # (Optional) Compute ratios as in the C++ code
                                            ratio_a = omega_cc_A / omega_cc_B if omega_cc_B != 0 else None
                                            ratio_b = omega_cs_A / omega_cs_B if omega_cs_B != 0 else None

                                            # Write summary file (append mode)
                                            '''
                                            write_summary(
                                                count=count,
                                                gam1=GAMMA_A,
                                                gam2=GAMMA_B,
                                                zetas1=ZETAS_A,
                                                zetas2=ZETAS_B,
                                                omega_cc1=omega_cc_A,
                                                omega_cc2=omega_cc_B,
                                                omega_cs1=omega_cs_A,
                                                omega_cs2=omega_cs_B,
                                                alpha=ALPHA,
                                                xi=XI,
                                                zetaQ=ZETAQ_A  # Using ZETAQ_A as representative
                                            )
                                            '''

                                            count += 1

    print("Computation done.")


if __name__ == '__main__':
    # Optionally seed the random number generator
    random.seed()  # Uses system time or os.urandom()
    main()

