/--------------------------------------------------------------/

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
/--------------------------------------------------------------/

handDetection
=============

Project for IUtil class of HEIG-VD.

Prerequists
-------------
* Python 2.7
* OpenCV 2.4.6.1

Usage
-----
On linux:
* $ python HDproject.py [debug]

To stop the program, please use the 'q' key, then on the terminal choose if you want to save or not the modifications

Files description
-----------------
* HDproject.py: Main Class
* faceDetection.py: Face detection class using haar. Threaded in main class
* .config: Config file
* haar: folder containing some haar XML