package main

import (
	"fmt"
	"math"
	"time"
)

type Position struct {
	lat, lon float64
	time     float64
}

type Stations struct {
	x1, x2, x3 *Position
}

func NewPosition(lat, lon, time float64) *Position {
	return &Position{math.Mod(lat, 2*math.Pi), math.Mod(lon, 2*math.Pi), time}
}

func NewPositionInDeg(lat, lon, time float64) *Position {
	return NewPosition(lat*math.Pi/180, lon*math.Pi/180, time)
}

func (p *Position) String() string {
	return fmt.Sprint("(", p.lat*180/math.Pi, ", ", p.lon*180/math.Pi, ")")
}

func f(x1, x2, s *Position) float64 {
	x := x1.d(s)/x2.d(s) - x1.time/x2.time
	return x
}

func (stations *Stations) F1(s *Position) float64 {
	return f(stations.x1, stations.x2, s)
}

func (stations *Stations) F2(s *Position) float64 {
	return f(stations.x2, stations.x3, s)
}

func derivative_in_first(fn func(s *Position) float64, pos *Position) float64 {
	h := 1e-6
	return (fn(NewPosition(pos.lat+h, pos.lon, pos.time)) - fn(NewPosition(pos.lat-h, pos.lon, pos.time))) / (2 * h)
}

func derivative_in_second(fn func(s *Position) float64, pos *Position) float64 {
	h := 1e-6
	return (fn(NewPosition(pos.lat, pos.lon+h, pos.time)) - fn(NewPosition(pos.lat, pos.lon-h, pos.time))) / (2 * h)
}

func (a *Position) d(b *Position) float64 {
	d := 6378000 * math.Acos(math.Cos(a.lat)*math.Cos(b.lat)*math.Cos(a.lon-b.lon)+math.Sin(b.lat)*math.Sin(a.lat))
	return d
}

func getCorrection(stations *Stations, position *Position) *Position {
	f := []float64{stations.F1(position), stations.F2(position)}
	J := []float64{
		derivative_in_first(stations.F1, position),
		derivative_in_second(stations.F1, position),
		derivative_in_first(stations.F2, position),
		derivative_in_second(stations.F2, position),
	}
	detJ := 1 / (J[0]*J[3] - J[1]*J[2])
	invJ := []float64{
		detJ * J[3],
		-detJ * J[1],
		-detJ * J[2],
		detJ * J[0],
	}
	return NewPosition(position.lat-(invJ[0]*f[0]+invJ[1]*f[1]), position.lon-(invJ[2]*f[0]+invJ[3]*f[1]), position.time)
}

func Solve(n int, stations *Stations, initPos *Position) *Position {
	betterPosition := initPos

	for i := 0; i < n; i++ {
		betterPosition = getCorrection(stations, betterPosition)
	}

	return betterPosition
}

func main() {
	start := time.Now()
	x1 := NewPositionInDeg(61.601944, -149.117222, 7.5)  // Palmer, Alaska
	x2 := NewPositionInDeg(39.746944, -105.210833, 23.0) // Golden, Colorado
	x3 := NewPositionInDeg(4.711111, -74.072222, 44.0)   // Bogota, Columbia

	stations := &Stations{x1, x2, x3}
	s := Solve(100000, stations, NewPositionInDeg(70.0, -180.0, 0))
	elapsed := time.Since(start)

	fmt.Println(s)
	fmt.Println("distances: ", x1.d(s)/1600, x2.d(s)/1600, x3.d(s)/1600)
	fmt.Println("time: ", elapsed)
}
