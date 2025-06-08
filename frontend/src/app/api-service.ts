import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = 'http://127.0.0.1:8000/api/predict';

  constructor(private http: HttpClient) {}

  predictMBTI(postText: string): Observable<any> {
    console.log('送出文字：', postText);
    return this.http.post(this.apiUrl, { post: postText });
  }
}
