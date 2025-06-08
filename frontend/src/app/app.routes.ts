import { Routes } from '@angular/router';
import { MbtiComponent } from './mbti/mbti.component';

export const routes: Routes = [
  { path: '', component: MbtiComponent },
  { path: 'mbti', component: MbtiComponent } 
];
