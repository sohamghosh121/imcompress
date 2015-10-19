/*
 * Options.h
 *
 *  Created on: 19 Oct 2015
 *      Author: sohamghosh
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

class Options {
private:
	double Mk = 0.4;
	double Mw = 0.2;
	double C = 1.0;
	int blockSize = 8;
	int M = 4;

public:
	Options();
	virtual ~Options();

	int getBlockSize() const {
		return blockSize;
	}

	void setBlockSize(int blockSize = 8) {
		this->blockSize = blockSize;
	}

	double getC() const {
		return C;
	}

	void setC(double c = 1.0) {
		C = c;
	}

	int getM() const {
		return M;
	}

	void setM(int m = 4) {
		M = m;
	}

	double getMk() const {
		return Mk;
	}

	void setMk(double mk = 0.4) {
		Mk = mk;
	}

	double getMw() const {
		return Mw;
	}

	void setMw(double mw = 0.2) {
		Mw = mw;
	}
};

#endif /* OPTIONS_H_ */
