#!/usr/bin/gnuplot -persist
#
#    
#    	G N U P L O T
#    	Version 4.2 patchlevel 6 
#    	last modified Sep 2009
#    	System: Linux 2.6.32-55-generic
#    
#    	Copyright (C) 1986 - 1993, 1998, 2004, 2007 - 2009
#    	Thomas Williams, Colin Kelley and many others
#    
#    	Type `help` to access the on-line reference manual.
#    	The gnuplot FAQ is available from http://www.gnuplot.info/faq/
#    
#    	Send bug reports and suggestions to <http://sourceforge.net/projects/gnuplot>
#    
unset clip points
set clip one
unset clip two
set bar 1.000000
set xdata
set ydata
set zdata
set x2data
set y2data
set boxwidth
set style fill  empty border
set dummy x,y
set format x "% g"
set format y "% g"
set format x2 "% g"
set format y2 "% g"
set format z "% g"
set format cb "% g"
set angles radians
set grid nopolar
set grid xtics mxtics ytics mytics noztics nomztics \
 nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics
set grid layerdefault   linetype 0 linewidth 1.000,  linetype 0 linewidth 1.000
set key title ""
set key off
unset label
unset arrow
unset style line
unset style arrow
unset logscale
set logscale x 10
set logscale y 10
set offsets 0, 0, 0, 0
set pointsize 1
set encoding default
unset polar
unset parametric
unset decimalsign
set view 60, 30, 1, 1  
set samples 1000, 1000
set isosamples 10, 10
set surface
unset contour
set clabel '%8.3g'
set mapping cartesian
set datafile separator whitespace
unset hidden3d
set cntrparam order 4
set cntrparam linear
set cntrparam levels auto 5
set cntrparam points 5
set size ratio -1 1,1
set origin 0,0
set style data lines
set style function lines
set xzeroaxis linetype -2 linewidth 1.000
set yzeroaxis linetype -2 linewidth 1.000
set x2zeroaxis linetype -2 linewidth 1.000
set y2zeroaxis linetype -2 linewidth 1.000
set ticslevel 0.5
set mxtics 10
set mytics 10
set mztics default
set mx2tics default
set my2tics default
set mcbtics default
set xtics autofreq
set ytics autofreq
set ztics autofreq
set nox2tics
set noy2tics
set cbtics autofreq
set title "Empirical Roofline Graph (Results.Local.Server/Run.001)" 
set timestamp bottom 
set timestamp "" 
set rrange [ * : * ] noreverse nowriteback  # (currently [0.00000:10.0000] )
set trange [ * : * ] noreverse nowriteback  # (currently [-5.00000:5.00000] )
set urange [ * : * ] noreverse nowriteback  # (currently [-5.00000:5.00000] )
set vrange [ * : * ] noreverse nowriteback  # (currently [-5.00000:5.00000] )
set xlabel "FLOPs / Byte" 
set x2label "" 
set xrange [1.000000e-02 : 1.000000e+02] noreverse nowriteback
set x2range [ * : * ] noreverse nowriteback  # (currently [-10.0000:10.0000] )
set ylabel "GFLOPs / sec" 
set y2label "" 
set yrange [1.000000e-01 : *] noreverse nowriteback
set y2range [ * : * ] noreverse nowriteback  # (currently [-10.0000:10.0000] )
set zlabel "" 
set zrange [ * : * ] noreverse nowriteback  # (currently [-10.0000:10.0000] )
set cblabel "" 
set cbrange [ * : * ] noreverse nowriteback  # (currently [-10.0000:10.0000] )
set zero 1e-08
set lmargin  -1
set bmargin  -1
set rmargin  -1
set tmargin  -1
set locale "C"
set pm3d explicit at s
set pm3d scansautomatic
set palette positive nops_allcF maxcolors 0 gamma 1.5 color model RGB 
set palette rgbformulae 7, 5, 15
set colorbox default
set loadpath 
set fit noerrorvariables

set term postscript solid color rounded
set output "roofline.ps"

# Plotting goes after this...
# Points

set object circle at first 0.074451,0.174285 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2
set object circle at first 0.071332,0.223071 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2
set object circle at first 0.061943,0.197753 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2
set object circle at first 0.074724,0.177562 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2
set object circle at first 0.073135,0.297467 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2
set object circle at first 0.069005,0.311343 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2
set object circle at first 0.074972,0.204909 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2
set object circle at first 0.074743,0.385036 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2
set object circle at first 0.074184,0.65275 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2
set object circle at first 0.074997,0.206917 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2
set object circle at first 0.074961,0.404275 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2
set object circle at first 0.074893,0.772657 radius char 0.5 \
    fillstyle empty border lc rgb '#aa1100' lw 2    

set label '64.8 GFLOPs/sec (FP64 Maximum)' at 2.0000000e+00,7.7712000e+01 left textcolor rgb '#000080'
set label 'L1 - 441.1 GB/s' at 3.0709191e-02,1.6392305e+01 left rotate by 45 textcolor rgb '#800000'
set label 'L2 - 55.6 GB/s' at 8.6462728e-02,5.8220975e+00 left rotate by 45 textcolor rgb '#800000'
set label 'L3 - 42.0 GB/s' at 1.2048394e-01,4.1781041e+00 left rotate by 45 textcolor rgb '#800000'
set label 'DRAM - 31.1 GB/s' at 1.2332698e-01,3.1728670e+00 left rotate by 45 textcolor rgb '#800000'
plot \
     (x <= 1.4679814e-01 ? 4.4115000e+02 * x : 1/0) lc 1 lw 2,\
     (x <= 1.1637017e+00 ? 5.5650000e+01 * x : 1/0) lc 1 lw 2,\
     (x <= 1.5433746e+00 ? 4.1960000e+01 * x : 1/0) lc 1 lw 2,\
     (x <= 2.0803084e+00 ? 3.1130000e+01 * x : 1/0) lc 1 lw 2,\
     (x >= 1.4679814e-01 ? 6.4760000e+01 : 1/0) lc 3 lw 2, \
     




